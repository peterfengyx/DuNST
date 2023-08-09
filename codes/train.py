# coding=utf-8
import logging
import os

import datasets
import transformers

from config import train_args, extra_args, GPT2Config
from dataset import build_dataset, DataCollatorForCVAE
from trainer import NLGTrainer
from graphs import GPT2VAE
import utils

logger = logging.getLogger(__name__)


def reload_ckpt():
    # Detecting last checkpoint.
    last_ckpt = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_ckpt = transformers.trainer_utils.get_last_checkpoint(train_args.output_dir)
        if last_ckpt is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_ckpt is not None and train_args.resume_from_checkpoint is None:
            print("reusing pretrained ckpt!")
            logger.info(
                f"Checkpoint detected, resuming training at {last_ckpt}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_ckpt


def main():

    utils.set_logger(logger, train_args)
    utils.set_seed(train_args.seed)
    tokenizer = utils.loadTokenizer(extra_args.tokenizer_dir)

    config = GPT2Config(vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    
    model = GPT2VAE(config)
    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Total ={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    #model = model.from_pretrained("../unilm/", config=config)
    #model = utils.load_plm(model, "../unilm/pytorch_model.bin")
    model = model.from_pretrained("../dvae-4089",config=config)
    #model = model.from_pretrained("../36569", config=config)
    

    #------------------------------
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    training_dataset, validation_dataset, unlabelled_dataset = build_dataset(train_args, extra_args, tokenizer)
    data_collator = DataCollatorForCVAE(tokenizer=tokenizer, padding='longest')
    training_dataset=training_dataset.select(range(1125))

    #return training_dataset, validation_dataset
    trainer = NLGTrainer(
        model=model,
        args=train_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics,
    )
    # Training
    #last_ckpt = reload_ckpt()
    train_result = trainer.train(#resume_from_checkpoint=last_ckpt,
        unlabelled_dataset=unlabelled_dataset,
        loss_names=['gen_loss', 'cl_loss', 'gen_kl', 'cl_kl', 'bow_loss', 'total_loss', 'accu', 'f1']
        )
    
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
