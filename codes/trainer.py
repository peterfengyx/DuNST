from re import T
from transformers.integrations import hp_params
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import Seq2SeqTrainer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.configuration_utils import PretrainedConfig
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_pt_utils import IterableDatasetShard, nested_detach
from transformers.trainer_callback import TrainerState
from transformers.deepspeed import deepspeed_init
from transformers import __version__
from transformers.utils import logging

from layers import NoiseLayer

from transformers.trainer_pt_utils import (
        DistributedLengthGroupedSampler,
        DistributedSamplerWithLoop,
        DistributedTensorGatherer,
        IterableDatasetShard,
        LabelSmoother,
        LengthGroupedSampler,
        SequentialDistributedSampler,
        ShardSampler
        )
from transformers.training_args import OptimizerNames, ParallelMode
from torch.utils.data import RandomSampler,SequentialSampler,WeightedRandomSampler

from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
)

from transformers.trainer_utils import (
    set_seed,
    has_length,
    get_last_checkpoint,
    speed_metrics,
    ShardedDDPOption,
    TrainOutput,
    PredictionOutput
)

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import (
        smp_forward_backward,
    )

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import utils
from config import extra_args

from datasets import load_metric
import torch.nn.functional as F
import datasets
metric = load_metric('bleu')

import collections
import math
import warnings
import sys
import time
import os
from tqdm.auto import tqdm

from gen_utils import GenerationTool

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    import optuna
    
TRAINER_STATE_NAME = "trainer_state.json"

#------------------------------------------------------------
class NLGTrainer(Seq2SeqTrainer):

    def train(
        self,
        loss_names,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        unlabelled_dataset = None,
        **kwargs,
    ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None and not is_sagemaker_mp_enabled():
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if unlabelled_dataset is not None:
            self.unlabelled_dataset=unlabelled_dataset
            unlabelled_dataloader =self.get_unlabelled_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        #if train_dataset_is_sized:
        if False:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial.assignments) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()
        self.train_loader2=self.get_train_dataloader()#copy.deepcopy(train_dataloader)
        # --------------------------------------------------------------------------
        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._tr_loss_dic = {}
        for loss_name in loss_names:
            self._tr_loss_dic[loss_name] = 0.0
        # --------------------------------------------------------------------------
        
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step_dic = self.training_step(model, inputs)
                else:
                    tr_loss_step_dic = self.training_step(model, inputs)

                # --------------------------------------------------------------------------
                for loss_name, loss_value in tr_loss_step_dic.items():
                    if args.logging_nan_inf_filter and (torch.isnan(loss_value) or torch.isinf(loss_value)):
                        self._tr_loss_dic[loss_name] +=  self._tr_loss_dic[loss_name] / (
                            1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        self._tr_loss_dic[loss_name] += loss_value.item()
                # --------------------------------------------------------------------------
                

                #self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise
                            nn.utils.clip_grad_norm_(model.parameters(),
                                args.max_grad_norm)

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    
                    
                    self.state.global_step += 1
                    
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                #if self.state.global_step>2000:
                #if False:
                #if float(torch.exp(tr_loss_step_dic['gen_loss']))<100
                self.self_training_steps_cls=extra_args.self_training_steps_cls
                self.self_training_steps_gen=extra_args.self_training_steps_gen
                self.supervised_steps=extra_args.supervised_steps
                self.train_cycle=self.self_training_steps_cls+self.self_training_steps_gen+self.supervised_steps
                
                if self.state.global_step>=extra_args.st_start_step and self.state.global_step%self.train_cycle==extra_args.st_start_step%self.train_cycle:
                    #add self-training after each epoch
                    print("Self Training:\n")
                    pseudo_list=[]
                    if extra_args.soft_label==True and extra_args.use_prior==False:
                        #sample soft data from generator
                        self.each_training_step=1
                        self.gen_num_batch=int(self.self_training_steps_gen//self.each_training_step)
                        print("Generating soft pseudo data:\n")
                        cnt=0
                        for u_step, u_inputs in enumerate(self.train_loader2):
                            pseudo_inputs=self.get_soft_label(model, inputs=u_inputs, temperature=extra_args.gen_temperature)
                            pseudo_list.append(pseudo_inputs)
                            cnt+=1
                            if cnt>=self.self_training_steps_gen:
                                break
                        if cnt<self.self_training_steps_gen:
                            for u_step, u_inputs in enumerate(unlabelled_dataloader):
                                pseudo_inputs=self.get_soft_label(model, inputs=u_inputs, temperature=extra_args.gen_temperature)
                                pseudo_list.append(pseudo_inputs)
                                cnt+=1
                                if cnt>=self.self_training_steps_gen:
                                    break
                        print(cnt*self.args.train_batch_size," Soft label generated.")

                    else:
                        #generate pseudo data from labeled/unlabeled data
                        self.each_training_step=1
                        self.gen_num_batch=int(self.self_training_steps_gen//self.each_training_step)
                        
                        print("Generating pseudo data:\n")
                        pseudo_list=[]
                        for i in range(self.gen_num_batch):
                            pseudo_inputs=self.gen_pseudo_data_from_label(model, num_labels=2, 
                                                                          num_per_class=self.args.train_batch_size//2, 
                                                                          temperature=extra_args.gen_temperature,
                                                                          return_logit=extra_args.soft_label)
                            pseudo_list.append(pseudo_inputs)
                            #pseudo_inputs=self.gen_pseudo_data_from_input(model, num_per_class=2, inputs=inputs)
                        print(self.gen_num_batch*self.args.train_batch_size," Pseudo data generated.")
                    
                    
                    
                    
                    if unlabelled_dataset is not None:
                        if extra_args.st_sampling in ["topk", "sample"]:
                            #Select top-k samples based on confidence
                            pseudo_l_list=[]
                            pseudo_dataset={'confidence':torch.tensor([])}
                            print("Annotating Pseudo Labels:")
                            for u_step, u_inputs in enumerate(unlabelled_dataloader):
                                pseudo_l_inputs=self.gen_pseudo_label_from_data(model, inputs=u_inputs, st_select=extra_args.st_select)
                                #pseudo_dataset['confidence'].to(pseudo_l_inputs['confidence'])
                                pseudo_dataset['confidence']=torch.cat([pseudo_dataset['confidence'],pseudo_l_inputs['confidence'].to("cpu")])
                                #for key in pseudo_l_inputs.keys():
                                #    pseudo_dataset[key]=torch.cat([pseudo_dataset[key],pseudo_l_inputs[key]])
                                for i in range(len(pseudo_l_inputs['confidence'])):
                                    temp_data={}
                                    for key in u_inputs.keys():
                                        temp_data[key]=pseudo_l_inputs[key][i].tolist()
                                    temp_data['cl_label']=temp_data['cl_labels']
                                    del temp_data['cl_labels']
                                    pseudo_l_list.append(temp_data)
                            
                            print(len(unlabelled_dataloader)*self.args.train_batch_size," Pseudo Labels Annotated.")
                            
                            if extra_args.st_sampling == "topk":
                                ids=torch.topk(pseudo_dataset['confidence'], self.self_training_steps_cls*self.args.train_batch_size,0, sorted=True).indices.reshape(-1)
                            elif extra_args.st_sampling == "sample":
                                ids=torch.multinomial(pseudo_dataset['confidence'].reshape(-1), self.self_training_steps_cls*self.args.train_batch_size, replacement=False)
                            del pseudo_dataset['confidence']
                            
                            pseudo_dataset=[]
                            num_pos=0
                            for i in range(len(ids)):
                                #temp_id=ids[i*self.args.train_batch_size:(i+1)*self.args.train_batch_size]
                                pseudo_dataset.append(pseudo_l_list[int(ids[i])])
                                num_pos+=pseudo_l_list[int(ids[i])]['cl_label']
                            pseudo_loader=DataLoader(pseudo_dataset,batch_size=self.args.train_batch_size,collate_fn=self.data_collator)
                            del pseudo_l_list
                            pseudo_l_list=[]
                            for u_step, u_inputs in enumerate(pseudo_loader):
                                pseudo_l_list.append(u_inputs)
                            
                            print("Top-",len(ids)," Sampled.")
                            print(num_pos," positive samples.")
                            with open("log", 'a') as f:
                                f.write(str(self.state.global_step))
                                f.write(" Step ")
                                f.write(str(num_pos))
                                f.write(" positive samples.\n")
                            
                            
                        else:
                            pseudo_l_list=[]
                            print("Annotating Pseudo Labels:")
                            for u_step, u_inputs in enumerate(unlabelled_dataloader):
                                if u_step>=self.self_training_steps_cls:
                                    break
                                pseudo_l_inputs=self.gen_pseudo_label_from_data(model, inputs=u_inputs, st_select=extra_args.st_select)
                                pseudo_l_list.append(pseudo_l_inputs)
                            print(self.self_training_steps_cls*self.args.train_batch_size," Pseudo Labels Annotated.")
                        
                        
                    
                    model.train()
                    for i in range(self.gen_num_batch):
                        pseudo_inputs=pseudo_list[i]
                        for st_step in range(self.each_training_step):
                            if step % args.gradient_accumulation_steps == 0:
                                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                            if (
                                ((step + 1) % args.gradient_accumulation_steps != 0)
                                and args.local_rank != -1
                                and args._no_sync_in_gradient_accumulation
                            ):
                                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                                with model.no_sync():
                                    tr_loss_step_dic = self.training_step(model, pseudo_inputs)
                            else:
                                tr_loss_step_dic = self.training_step(model, pseudo_inputs)

                            # --------------------------------------------------------------------------
                            
                            # --------------------------------------------------------------------------
                            

                            #self.current_flos += float(self.floating_point_ops(inputs))

                            # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                            if self.deepspeed:
                                self.deepspeed.step()
                            
                            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                                # last step in epoch but step is always smaller than gradient_accumulation_steps
                                steps_in_epoch <= args.gradient_accumulation_steps
                                and (step + 1) == steps_in_epoch
                            ):
                                # Gradient clipping
                                if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                                    # deepspeed does its own clipping

                                    if hasattr(self.optimizer, "clip_grad_norm"):
                                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                        self.optimizer.clip_grad_norm(args.max_grad_norm)
                                    elif hasattr(model, "clip_grad_norm_"):
                                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                        model.clip_grad_norm_(args.max_grad_norm)
                                    else:
                                        # Revert to normal clipping otherwise
                                        nn.utils.clip_grad_norm_(model.parameters(),
                                            args.max_grad_norm)

                                # Optimizer step
                                optimizer_was_run = True
                                if self.deepspeed:
                                    pass  # called outside the loop
                                else:
                                    self.optimizer.step()

                                if optimizer_was_run and not self.deepspeed:
                                    self.lr_scheduler.step()

                                model.zero_grad()
                                
                                self.state.global_step += 1
                                #self.state.epoch = epoch + (step + 1) / steps_in_epoch
                                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                                #self._maybe_log_save_evaluate(model, trial, epoch, ignore_keys_for_eval)
                                if self.control.should_epoch_stop or self.control.should_training_stop:
                                    break
                    self._globalstep_last_logged+=self.self_training_steps_gen #ignore pseudo data for evaluation
                    
                    
                    if unlabelled_dataset is not None:
                        model.train()
                        for i in range(self.self_training_steps_cls):
                            pseudo_l_inputs=pseudo_l_list[i]
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                            if (
                                ((u_step + 1) % args.gradient_accumulation_steps != 0)
                                and args.local_rank != -1
                                and args._no_sync_in_gradient_accumulation
                            ):
                                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                                with model.no_sync():
                                    tr_loss_step_dic = self.training_step(model, pseudo_l_inputs)
                            else:
                                tr_loss_step_dic = self.training_step(model, pseudo_l_inputs)
                            
                            
                            if self.deepspeed:
                                self.deepspeed.step()
                            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                                # last step in epoch but step is always smaller than gradient_accumulation_steps
                                steps_in_epoch <= args.gradient_accumulation_steps
                                and (step + 1) == steps_in_epoch
                            ):
                                # Gradient clipping
                                if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                                    # deepspeed does its own clipping

                                    if hasattr(self.optimizer, "clip_grad_norm"):
                                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                        self.optimizer.clip_grad_norm(args.max_grad_norm)
                                    elif hasattr(model, "clip_grad_norm_"):
                                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                        model.clip_grad_norm_(args.max_grad_norm)
                                    else:
                                        # Revert to normal clipping otherwise
                                        nn.utils.clip_grad_norm_(model.parameters(),
                                            args.max_grad_norm)

                                # Optimizer step
                                optimizer_was_run = True
                                if self.deepspeed:
                                    pass  # called outside the loop
                                else:
                                    self.optimizer.step()

                                if optimizer_was_run and not self.deepspeed:
                                    self.lr_scheduler.step()

                                model.zero_grad()
                                
                                self.state.global_step += 1
                                #self.state.epoch = epoch + (step + 1) / steps_in_epoch
                                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                                #self._maybe_log_save_evaluate(model, trial, epoch, ignore_keys_for_eval)
                                if self.control.should_epoch_stop or self.control.should_training_stop:
                                    break
                        self._globalstep_last_logged+=self.self_training_steps_cls
                    print("Self-training over. Steps=", self.state.global_step)

          
            
            if self.control.should_training_stop:
                break
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(model, trial, epoch, ignore_keys_for_eval)


            

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        #self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        
        # --------------------------------------------------------------------------
        for loss_name, loss_value in self._tr_loss_dic.items():
            metrics[loss_name] = loss_value / self.state.global_step
        # --------------------------------------------------------------------------

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, metrics['total_loss'], metrics)
    
    #---------------------------------------------------------
    def get_train_dataloader(self, specified_bsz=None) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        bsz = self.args.train_batch_size if specified_bsz is None else specified_bsz

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=bsz,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=bsz,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=bsz,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    #unlabelled dataloader sampling
    def _get_unlabelled_sampler(self, weight=None, num_samples=None) -> Optional[torch.utils.data.Sampler]:
        #from transformers.utils.import_utils import is_datasets_available
        if self.unlabelled_dataset is None or not has_length(self.unlabelled_dataset):
            return None

        generator = None
        if self.args.world_size <= 1: #and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if isinstance(self.unlabelled_dataset, datasets.Dataset): #and is_datasets_available()
                lengths = (
                    self.unlabelled_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.unlabelled_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.unlabelled_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.unlabelled_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if weight==None:
                    return RandomSampler(self.unlabelled_dataset, generator=generator)
                else:
                    return WeightedRandomSampler(weights=weight,num_samples=num_samples,replacement=False, generator=generator)
                #if True:#_is_torch_generator_available:
                #    return RandomSampler(self.unlabelled_dataset, generator=generator)
                #return RandomSampler(self.unlabelled_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.unlabelled_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.unlabelled_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            
    def get_unlabelled_dataloader(self, specified_bsz=None, weight=None, num_samples=None) -> DataLoader:

        if self.unlabelled_dataset is None:
            raise ValueError("Trainer: training requires a unlabelled_dataset.")

        #train_dataset = self.train_dataset
        train_dataset = self.unlabelled_dataset
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        bsz = self.args.train_batch_size if specified_bsz is None else specified_bsz
        
        '''
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=bsz,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=bsz,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        '''

        train_sampler = self._get_unlabelled_sampler(weight=weight,num_samples=num_samples)

        return DataLoader(
            train_dataset,
            batch_size=bsz,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_soft_label(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        #num_per_class: int,
        temperature: float=5.0
    ):
        model.eval()
        inp_dic = self._prepare_inputs(inputs)
        outp_dic = inp_dic
        #results = self.predict(testing_set, metric_key_prefix="predict")
        with torch.no_grad():
            outs=model(inp_dic)
            logit=outs['gen_logits'].detach()
            bos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
            beginer = torch.tensor([bos_idx]).to(inp_dic['seq_attn_mask'])
            #beginer = model.transformer.wte(beginer).tile(outs['gen_logits'].shape[0],1,1)
            beginer = F.one_hot(beginer, num_classes=logit.shape[-1]).tile(outs['gen_logits'].shape[0],1,1)
            #print(beginer.shape)
            outp_dic['past_logit']=torch.cat([beginer,logit[:,:-1,:]], dim=1)
        #outp_dic['past_logit']=model.get_logit(inp_dic)
        #outp_dic['seq_attn_mask']=inp_dic['seq_attn_mask'][:,1:]
        #outp_dic['seq_token_type_ids']=inp_dic['seq_token_type_ids'][:,1:]

        return outp_dic

    def gen_pseudo_label_from_data(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        #num_per_class: int,
        num_labels: int=2,
        st_select: str="argmax",
        mc_dropout: int=0
    ):
        model.eval()
        inp_dic = self._prepare_inputs(inputs)
        outp_dic = inp_dic
        #results = self.predict(testing_set, metric_key_prefix="predict")
        loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False)
        
        probs = F.softmax(logits,dim=-1)
        if mc_dropout==0:
            outp_dic['confidence']= torch.max(probs,-1).values.unsqueeze(-1)
        else:
            assert mc_dropout>0
            #MC-dropout
            utils.enable_dropout(model)
            output_list=[]
            for _ in range(mc_dropout):
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False)
                probs = F.softmax(logits,dim=-1)
                output_list.append(probs.unsqueeze(1)) #[bsz,1, class]
            output_list=torch.cat(output_list,dim=1)
            mean=torch.mean(output_list,dim=1) #[bsz, class]
            var=torch.var(output_list,dim=1)
            epsilon=sys.float_info.min
            entropy=-torch.sum(mean*torch.log(mean+epsilon),dim=-1)
            mutual_info=entropy-torch.mean(torch.sum(-output_list*torch.log(output_list+epsilon),dim=-1), dim=1) #[bsz]
            outp_dic['confidence']= mutual_info.unsqueeze(-1)
        
        if st_select=="argmax":
            preds = torch.argmax(logits, dim=-1)
            outp_dic['cl_labels']=preds
        elif st_select=="sample":
            preds=torch.multinomial(probs,num_samples=1)
            outp_dic['cl_labels']=preds
            
        
        return outp_dic
    
    #-------------------------------------------------------
    def gen_pseudo_data_from_label(
        self,
        model: nn.Module,
        num_per_class: int,
        num_labels: int=2,
        temperature: float=0.25,
        return_logit: bool=False
    ):# -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        num_type=num_labels
        generator = GenerationTool()
        model.eval()
        
        outp_dic = {}
                
        #bos_token_idx = self.tokenizer.vocab['[L2R_SEP]']
        eos_token = self.tokenizer.sep_token
        bos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        eos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        max_length = min(extra_args.max_body_len+extra_args.max_title_len+2, 512)
        seqs=[]
        titles=[]
        #p_labels=[]
        
        
        max_len_gen=0
        max_len = max_length
        
        
        beginer=torch.ones((num_per_class*num_labels,1),dtype=torch.long, device=model.device)*bos_idx
        att=torch.ones((num_per_class*num_labels,1),dtype=torch.long, device=model.device)
        token_type=torch.ones((num_per_class*num_labels,1),dtype=torch.long, device=model.device)
        pseudo_label=torch.cat([torch.zeros((num_per_class),dtype=torch.long, device=model.device),torch.ones((num_per_class),dtype=torch.long, device=model.device)],dim=0)
        z,_=model.get_gen_prior_z(None, pseudo_label,2)
        with torch.no_grad():
            outs = generator.generate(
                    model, beginer, z,
                    attention_mask=att,
                    token_type_ids=token_type,
                    bos_token_id=bos_idx,
                    do_sample=True, top_p=0.9, temperature=temperature, length_penalty=1.0,
                    repetition_penalty=1.0, no_repeat_ngram_size=4,
                    use_cache=True, max_length=max_len,
                    pad_token_id=pad_idx, eos_token_id=eos_idx,
                    output_scores=False, return_dict_in_generate=True)
        for i in range(num_per_class*num_labels):
            seq = outs['sequences'][i]
            #seqs.append(seq)
            #print(seq.shape)
            #seq = seq[int(sum(inp_dic['body_attn_mask'][i]==1)):].tolist()
            #print(sum(inp_dic['body_attn_mask'][i]==1))
            #print(len(seq))
            seq=seq.tolist()
            if eos_idx in seq:
                seq = seq[0:seq.index(eos_idx)]
            #print(len(seq))
            if len(seq)>max_len_gen:
                max_len_gen=len(seq)
            titles.append(torch.tensor(seq,dtype=torch.long, device=model.device))
        
        #p_labels.append(pseudo_label)
        #max_len_gen=max_len
        outp_dic['title_ids']=torch.zeros((len(titles),max_len_gen),dtype=torch.long, device=model.device)
        outp_dic['title_attn_mask']=torch.zeros((len(titles),max_len_gen),dtype=torch.long, device=model.device)
        outp_dic['title_token_type_ids']=torch.ones((len(titles),max_len_gen),dtype=torch.long, device=model.device)
        
        #outp_dic['cl_labels']=torch.cat(p_labels,dim=0)
        outp_dic['cl_labels']=pseudo_label
        for i in range(len(titles)):
            outp_dic['title_ids'][i,:len(titles[i])]=titles[i]
            outp_dic['title_attn_mask'][i,:len(titles[i])]=torch.ones(len(titles[i]),dtype=torch.long, device=model.device)
            
        outp_dic['seq_ids']=outp_dic['title_ids']
        outp_dic['seq_attn_mask']=outp_dic['title_attn_mask']
        outp_dic['seq_token_type_ids']=outp_dic['title_token_type_ids']
        if return_logit:
            logit=torch.cat(outs['scores'],dim=-1).reshape(len(titles),len(outs['scores']),-1)
            beginer = torch.tensor([bos_idx]).to(outp_dic['seq_attn_mask'])
            #beginer = model.transformer.wte(beginer).tile(outs['gen_logits'].shape[0],1,1)
            beginer = F.one_hot(beginer, num_classes=logit.shape[-1]).tile(len(titles),1,1)
            #print(beginer.shape)
            #print(outp_dic['seq_ids'].shape)
            #print(logit.shape)
            outp_dic['past_logit']=torch.cat([beginer,logit[:,:(max_len_gen-1),:]], dim=1)
        
        
        return outp_dic
    
    def gen_pseudo_data_from_input(
        self,
        model: nn.Module,
        num_per_class: int,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        ignore_keys: Optional[List[str]] = None,
    ):# -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        num_type=2
        generator = GenerationTool()
        model.eval()
        inp_dic = self._prepare_inputs(inputs)
        #print(inp_dic.keys())
        #print(inp_dic['body_ids'].shape)
        outp_dic = {}
        outp_dic['body_ids']=inp_dic['body_ids'].tile((num_type,1))#.reshape(-1,inp_dic['body_ids'].shape[-1])
        outp_dic['body_attn_mask']=inp_dic['body_attn_mask'].tile((num_type,1))#.reshape(-1,inp_dic['body_attn_mask'].shape[-1])
        outp_dic['body_token_type_ids']=inp_dic['body_token_type_ids'].tile((num_type,1))#.reshape(-1,inp_dic['body_token_type_ids'].shape[-1])
        
                
        bos_token_idx = self.tokenizer.vocab['[L2R_SEP]']
        eos_token = self.tokenizer.sep_token
        eos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        max_length = min(extra_args.max_body_len+extra_args.max_title_len+2, 512)
        seqs=[]
        titles=[]
        p_labels=[]
        
        for key in inp_dic.keys():
            inp_dic[key] = inp_dic[key].to(model.device)
        c_h = model.get_seq_rep(inp_dic['body_ids'],
            inp_dic['body_attn_mask'], inp_dic['body_token_type_ids'])
        #print(c_h.shape)
        #print('input_ids',inp_dic['body_ids'].shape)
        #print('seq_ids',inp_dic['seq_ids'].shape)
        #print('title_ids',inp_dic['title_ids'].shape)
        
        #print('seq_token_type_ids',inp_dic['seq_token_type_ids'][3])
        #print('seq_attn_mask',inp_dic['seq_attn_mask'][3])
        #print('atten',inp_dic['body_attn_mask'].shape)
        #print('body_token_type_ids',inp_dic['body_attn_mask'].shape)
        #raise('stop')
        
        max_len_gen=0
        max_len = min(inp_dic['body_ids'].size(-1) + int(inp_dic['title_ids'].size(-1) * 1.1), max_length)

        for j in range(num_type):
            pseudo_label=torch.ones((c_h.shape[0])).to(inp_dic['cl_labels'])*j
            z,_=model.get_gen_prior_z(c_h, pseudo_label,2)
            #print(z.shape)
            #raise('stop')
            with torch.no_grad():
                outs = generator.generate(
                        model, inp_dic['body_ids'], z,
                        attention_mask=inp_dic['body_attn_mask'],
                        token_type_ids=inp_dic['body_token_type_ids'],
                        do_sample=True, top_p=0.9, temperature=0.25, length_penalty=1.0,
                        repetition_penalty=1.0, no_repeat_ngram_size=4,
                        use_cache=True, max_length=max_len,
                        pad_token_id=pad_idx, eos_token_id=eos_idx,
                        output_scores=False, return_dict_in_generate=True)
            #print(outs['sequences'].shape)
            for i in range(len(inp_dic['body_ids'])):
                seq = outs['sequences'][i]
                seqs.append(seq)
                #print(seq.shape)
                seq = seq[int(sum(inp_dic['body_attn_mask'][i]==1)):].tolist()
                #print(sum(inp_dic['body_attn_mask'][i]==1))
                #print(len(seq))
                if eos_idx in seq:
                    seq = seq[0:seq.index(eos_idx)]
                #print(len(seq))
                if len(seq)>int(inp_dic['title_ids'].size(-1) * 2):
                    seq = seq[0:int(inp_dic['title_ids'].size(-1) * 2)]
                if len(seq)>max_len_gen:
                    max_len_gen=len(seq)
                titles.append(torch.tensor(seq).to(inp_dic['body_ids']))
            p_labels.append(pseudo_label)
        
        outp_dic['title_ids']=torch.zeros((len(seqs),max_len_gen)).to(inp_dic['body_ids'])
        outp_dic['title_attn_mask']=torch.zeros((len(seqs),max_len_gen)).to(inp_dic['body_ids'])
        outp_dic['title_token_type_ids']=torch.ones((len(seqs),max_len_gen)).to(inp_dic['body_ids'])
        
        outp_dic['cl_labels']=torch.cat(p_labels,dim=0)
        
        
        body_len=outp_dic['body_ids'].shape[-1]
        seq_len=min(body_len+max_len_gen,512)
        outp_dic['seq_ids']=torch.zeros((len(seqs),seq_len)).to(inp_dic['body_ids'])
        outp_dic['seq_attn_mask']=torch.zeros((len(seqs),seq_len)).to(inp_dic['body_ids'])
        outp_dic['seq_token_type_ids']=torch.ones((len(seqs),seq_len)).to(inp_dic['body_ids'])
        #outp_dic['seq_token_type_ids']=torch.cat([outp_dic['body_token_type_ids'],outp_dic['title_token_type_ids'][:,:(seq_len-body_len)]],dim=1)
        
        #outp_dic['seq_ids'][:,:body_len]=outp_dic['body_ids']
        
        
        for i in range(len(seqs)):
            outp_dic['title_ids'][i,:len(titles[i])]=titles[i]
            outp_dic['title_attn_mask'][i,:len(titles[i])]=torch.ones(len(titles[i])).to(inp_dic['body_ids'])
            
            outp_dic['seq_ids'][i,:len(seqs[i])]=seqs[i][:seq_len]
            outp_dic['seq_attn_mask'][i,:len(seqs[i])]=torch.ones(min(len(seqs[i]),seq_len)).to(inp_dic['body_ids'])
            outp_dic['seq_token_type_ids'][i,:int(sum(outp_dic['body_attn_mask'][i]==1))]= torch.zeros(int(sum(outp_dic['body_attn_mask'][i]==1))).to(inp_dic['body_ids'])
            
        '''
        outp_dic['seq_ids'] = [ body[0:-1] + [bos_token_idx] + title[1:] for body, title in zip(body_inputs['input_ids'], title_inputs['input_ids'])]
        outp_dic['seq_mask'] = [ body + title[1:] for body, title in zip(body_inputs['attention_mask'], title_inputs['attention_mask'])]
        outp_dic['seq_type_ids'] = [ [0] * len(body) + [1]*len(title[1:]) for body, title in zip(body_inputs['input_ids'], title_inputs['input_ids'])]
        '''
        #print('input_ids',outp_dic['body_ids'].shape)
        #print('seq_ids',outp_dic['seq_ids'].shape)
        #print('title_ids',outp_dic['title_ids'].shape)
        
        
        return outp_dic

            
    
    
    #-------------------------------------------------------
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        

        labels = nested_detach(inputs.get('cl_labels').unsqueeze(1))
        if len(labels) == 1:
            labels = labels[0]


        with torch.no_grad():

            loss,gen_loss, logits = self.compute_loss(model, inputs, valid=True)
            loss = loss.detach()
            gen_loss = gen_loss.detach()
            logits = logits.detach()

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (gen_loss, logits, labels)
    
    def prediction_step_mc(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        

        labels = nested_detach(inputs.get('cl_labels').unsqueeze(1))
        if len(labels) == 1:
            labels = labels[0]


        with torch.no_grad():
            loss,gen_loss, logits = self.compute_loss(model, inputs, valid=True)
            loss = loss.detach()
            gen_loss = gen_loss.detach()
            logits = logits.detach()
            

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (gen_loss, logits, labels) 
    
    # ------------------------------------------------
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        if is_sagemaker_mp_enabled():
            scaler = None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        loss_dic = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            for loss_name, loss_value in loss_dic.items():
                loss_dic[loss_name] = loss_value.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            for loss_name, loss_value in loss_dic.items():
                loss_dic[loss_name] = loss_value / self.args.gradient_accumulation_steps
        
        loss = loss_dic['total_loss']
        
        if self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        for loss_name in loss_dic:
            loss_dic[loss_name].detach()
        
        return loss_dic


    #-----------------------------------------------------------
    # ----------------------------------------------------------
    def _maybe_log_save_evaluate(self, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            logs["learning_rate"] = self._get_learning_rate()
            
            logs["gen_kl scale"] = utils.getGenKLScale(self.state.global_step)
            logs["cl_kl scale"] = utils.getCLKLScale(self.state.global_step)
            
            for loss_name, loss_value in self._tr_loss_dic.items():
                # reset tr_loss to zero
                #logs[loss_name] = round(loss_value / self.state.global_step, 4)
                logs[loss_name] = round(loss_value / (self.state.global_step-self._globalstep_last_logged), 4)
                self._tr_loss_dic[loss_name]=0.0
            
            #self.store_flos()
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)
            
            if self.args.should_save:
                self.state.save_to_json(os.path.join(self.args.logging_dir, TRAINER_STATE_NAME))
        
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # ----------------------------------------------------------
    def compute_loss(self, model, inputs, valid=False):   
        
        outs = model(inputs, valid)
        #print(outs.keys())
        if valid:
            return outs['cl_loss'],outs['gen_loss'], outs['cl_logits']
        
        #---------------------------------------------
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outs[self.args.past_index]
        
        gen_kl_weight = utils.getGenKLScale(self.state.global_step)
        cl_kl_weight = utils.getCLKLScale(self.state.global_step)
        #cl_weight=extra_args.cl_weight
        cl_weight=utils.getCLScale(self.state.global_step)
        if cl_weight==0:
            cl_kl_weight=0.0
        
        total_loss = outs['gen_loss'] + outs['cl_loss'] * extra_args.cl_weight + \
            outs['gen_kl'] * gen_kl_weight*0.1 + outs['cl_kl'] * cl_kl_weight + \
            outs['bow_loss'] * extra_args.bow_weight

        accu, f1 = utils.getAccuF1(outs['cl_logits'], inputs['cl_labels'])
        
        loss_dic = {'gen_loss':outs['gen_loss'], 'cl_loss': outs['cl_loss'],
            'gen_kl':outs['gen_kl'], 'cl_kl':outs['cl_kl'], 'bow_loss':outs['bow_loss'],
            'total_loss': total_loss, 'accu':accu, 'f1':f1}

        if self.should_print() and not valid:
            utils.print_samples(self.tokenizer,
                outs['gen_logits'], inputs['title_ids'], inputs['seq_token_type_ids'],
                self._tr_loss_dic, self.state.global_step,
                self._get_learning_rate(), gen_kl_weight, cl_kl_weight)
        
        return loss_dic
    
    # ------------------------
    def should_print(self):
        return  self.state.global_step > 0 and (self.state.global_step % extra_args.print_sample_steps == 0) \
            and (self.args.local_rank == -1 or self.args.local_rank == 0)

    
    def is_parallel_mode(self, model):
        return isinstance(model, torch.nn.DataParallel) or isinstance(model,torch.nn.parallel.DistributedDataParallel)
