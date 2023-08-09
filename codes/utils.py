import logging
import os
import sys
import copy
import random
import numpy as np
import math
import tqdm

import torch
import torch.nn.functional as F

import transformers
from transformers.generation_utils import top_k_top_p_filtering
from transformers import AutoTokenizer, EvalPrediction
import datasets

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from config import extra_args


def get_mapdic():
    # load map dic
    dic = {} # bert to ours
    with open("../mapdic", "r") as fin:
        for line in fin:
            para = line.strip().split("|")
            dic[para[0]] = para[1]

    new_dic = {}
    for i in range(1, 12):
        for k, v in dic.items():
            if "0" in k:
                new_k = copy.deepcopy(k).replace("0", str(i))
                new_v = copy.deepcopy(v).replace("0", str(i))
                new_dic[new_k] = new_v

    #print(dic)
    dic.update(new_dic)
    return dic

def load_plm(model, plm_dir):
    map_dic = get_mapdic() # bert to ours
    #for k,v in map_dic.items(): print(k, v)
    sub_dic = {}
    plm_ckpt = torch.load(plm_dir)

    for k, v in plm_ckpt.items():
        if k in map_dic:
            sub_dic[map_dic[k]] = v
    ori_dic = model.state_dict()
    print("load plm modules: %d" % (len(sub_dic)))
    ori_dic.update(sub_dic)
    model.load_state_dict(ori_dic)
    return model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True


def set_logger(logger, train_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}"
        + f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )


def print_samples(tokenizer, gen_logits, tgts, type_ids,
        loss_dic, global_steps, lr, gen_kl_weight, cl_kl_weight):

    # select print_sample_num samples
    sample_num = min(gen_logits.size(0), extra_args.print_sample_num)
    indices = random.sample(list(range(gen_logits.size(0))), sample_num)
    
    sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        
    for i, idx in enumerate(indices):
        
            
        tgt = tokenizer.decode(tgts[idx, :], skip_special_tokens=True)
        
        tgt_len = (1-type_ids[idx, :]).sum().item()
        filtered_logits = top_k_top_p_filtering(gen_logits[idx, tgt_len:, :], top_p=0.9)
        probs = filtered_logits.softmax(dim=-1)
        sampled_tokens = torch.multinomial(probs, 1, replacement=True)[:, 0]
        
        sampled_tokens = sampled_tokens.tolist()
        
        
        #print(gen_ids)
        if sep_idx in sampled_tokens:
            sampled_tokens = sampled_tokens[0:sampled_tokens.index(sep_idx)]
        
        gen = tokenizer.decode(sampled_tokens[0:extra_args.max_title_len], skip_special_tokens=True)
        
        print("target #%d: %s" % (i, tgt))
        print("generated seq #%d: %s" % (i, "#"+gen+"#"))

    gen_loss = loss_dic['gen_loss'] / (global_steps + 1)
    cl_loss = loss_dic['cl_loss'] / (global_steps + 1)
    bow_loss = loss_dic['bow_loss'] / (global_steps + 1)
    
    accu = loss_dic['accu'] / (global_steps + 1)
    f1 = loss_dic['f1'] / (global_steps + 1)
    
    gen_kl = loss_dic['gen_kl'] / (global_steps + 1)
    cl_kl = loss_dic['cl_kl'] / (global_steps + 1)

    print("\n")
    info_str = "Training gen loss: %.2f, cl loss: %.2f, bow loss: %.2f" + \
        "\n ppl: %.2f, accu: %.2f, f1: %.2f" + \
        "\n gen kl: %.2f, cl kl: %.2f" + \
        "\n lr: %.2f 1e-4, gen kl weight: %.4f, cl kl weight: %.4f \n"
    
    loss_info = info_str % (gen_loss, cl_loss, bow_loss, np.exp(gen_loss), accu*100, f1*100,
            gen_kl, cl_kl, lr * 1e4, gen_kl_weight, cl_kl_weight
    )
    
    print(loss_info)


def getAccuF1(logits, labels):
    cl_preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).detach().cpu().numpy()
    cl_labels = labels.detach().cpu().numpy()
    
    accu = accuracy_score(cl_labels, cl_preds)
    f1 = f1_score(cl_labels, cl_preds, zero_division=1)
    
    return torch.tensor(accu), torch.tensor(f1)


def getGenKLScale(cur_steps):
    if cur_steps > extra_args.gen_kl_annealing_steps:
        return 1.0
    period = extra_args.gen_kl_annealing_steps / extra_args.gen_kl_n_cycle
    cur_pos = cur_steps % period
    return min(1.0 / (period*extra_args.kl_cycle_ratio) * cur_pos, 1.0)


def getCLKLScale(cur_steps):
    if cur_steps > extra_args.cl_kl_annealing_steps:
        return 1.0
    period = extra_args.cl_kl_annealing_steps / extra_args.cl_kl_n_cycle
    cur_pos = cur_steps % period
    return min(1.0 / (period*extra_args.kl_cycle_ratio) * cur_pos, 1.0)
    

def getCLScale(cur_steps):
    if cur_steps < extra_args.cl_annealing_steps:
        return extra_args.cl_weight
    cl_scale = extra_args.cl_weight*(1-(cur_steps-extra_args.cl_annealing_steps)*extra_args.cl_decay)
    return max(cl_scale, 1.0)


def getLinearScale(cur_steps):
    if cur_steps < extra_args.attnl1_freeze_steps:
        attn_scale = 0.0
    elif cur_steps < extra_args.attnl1_freeze_steps+extra_args.attnl1_warm_steps:
        attn_scale = (cur_steps - extra_args.attnl1_freeze_steps) / extra_args.attnl1_warm_steps
    else:
        attn_scale = 1.0
    
    return attn_scale


def loadTokenizer(tokenizer_dir):
    if len(os.listdir(tokenizer_dir)) == 0:
        print("download tokenizer!")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=tokenizer_dir)
        tokenizer.save_pretrained(tokenizer_dir)
    else:
        print("load tokenizer from dir!")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
        
    return tokenizer


def compute_metrics(p: EvalPrediction):
    
    ori_preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(ori_preds, axis=1)
    def softmax(x,axis=1):
        max_x = np.max(x,axis=axis,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max_x) #subtracts each row with its max value
        sum_e = np.sum(e_x,axis=axis,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum_e 
        return f_x
    probs = softmax(ori_preds, axis=1)
    
    labels = np.concatenate(p.label_ids, axis=0)  

    accu = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs[:, -1])

    return {"accuracy": accu, "f1": f1, "auc":auc}

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)
    
def calc_iwnll(model, iters,device='cuda', ns=5):
    report_kl_loss = report_ce_loss = report_loss = 0
    report_num_words = report_num_sents = 0
    #for _,inputs in enumerate(tqdm(iters, desc="Evaluating PPL")):
    for i,inputs in enumerate(iters):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs['seq_ids']
        attention_mask = inputs['seq_attn_mask']
        seq_len = attention_mask.sum(-1).long()
        report_num_sents += input_ids.size(0)
        report_num_words += seq_len.sum().item()
        #kl_loss = model.get_klloss(input_ids, attention_mask)
        kl_loss = model.get_klloss(inputs)
        ll_tmp = []
        ce_tmp = []
        for _ in range(ns):
            log_gen, log_likelihood = model.iw_sample(inputs)
            ce_tmp.append(log_gen)
            ll_tmp.append(log_likelihood)

        ll_tmp = torch.stack(ll_tmp, dim=0)
        log_prob_iw = log_sum_exp(ll_tmp, dim=0) - math.log(ns)
        log_ce_iw = torch.mean(torch.stack(ce_tmp), dim=0)/seq_len
        report_kl_loss += kl_loss.sum().item()
        report_ce_loss += log_ce_iw.sum().item()
        report_loss += log_prob_iw.sum().item()
        if i%10==0:
            print(i)

    elbo = (report_kl_loss - report_ce_loss) / report_num_sents
    nll = - report_ce_loss / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(-report_loss / report_num_words)
    return ppl, elbo, nll, kl

def calc_au(model, iters, delta=0.2):
    """compute the number of active units
    """
    cnt = 0
    for inputs in tqdm(iters, desc="Evaluating Active Units, Stage 1"):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        mean, _ = model.get_encode_states(input_ids=input_ids, attention_mask=attention_mask)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for inputs in tqdm(iters, desc="Evaluating Active Units, Stage 2"):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        mean, _ = model.get_encode_states(input_ids=input_ids, attention_mask=attention_mask)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)
    au = (au_var >= delta).sum().item()
    au_prop = au / mean.size(-1)
    return au_prop


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
'''           
    def get_unsup_loss(self, batch, n_unsup):
        def get_baseline(R):
            return torch.mean(R).detach() # should use detach(), just a negative number for each reward
        label_source, unlabel_source = self.checkLabelSource(batch)
        batch_size, T = batch['word_idx'][label_source].size()
        decode_len = self.output[unlabel_source]['trans']['decode_len']
        
		# pass the gradient through the generated data between two forwards using REINFORCE
        reward = self.get_reward_from_rec(label_source, batch) # (B, beam_size)
        baseline = get_baseline(reward)
        rewardDiff = reward
        #		rewardDiff = reward - baseline # (B, T)
		
		# rl loss
        rec_loss, kl, kl_weight = 0, 0, 0
        assert self.output[unlabel_source]['trans']['mode'] == 'gen'
        logprobs = self.output[unlabel_source]['trans']['logprobs'] # (B, T) or (B, beam_size) in beam_search
        rl_loss = torch.mean(-logprobs*rewardDiff)

		# rec loss
		assert self.output2[label_source]['trans']['mode'] == 'teacher_force'
		index = batch['word_idx'][label_source].unsqueeze(1).repeat(1, self.config.beam_size, 1).view(batch_size*self.config.beam_size, T)
		rec_loss = NLLEntropy(self.output2[label_source]['trans']['logits'], index, ignore_idx=self.dataset.vocab[label_source]['<PAD>'])

		# auto-encoder loss
		auto_loss = NLLEntropy(self.output[label_source]['auto']['logits'], batch['word_idx'][label_source], \
								ignore_idx=self.dataset.vocab[label_source]['<PAD>'])

		# kl loss (B, Z) -> (beam_size*B, Z)
		self.unsup_mu[label_source] = self.unsup_mu[label_source].unsqueeze(1)\
						.repeat(1, self.config.beam_size, 1).view(batch_size*self.config.beam_size, self.config.latent_size)
		self.unsup_logvar[label_source] = self.unsup_logvar[label_source].unsqueeze(1)\
						.repeat(1, self.config.beam_size, 1).view(batch_size*self.config.beam_size, self.config.latent_size)
		kl_loss = 0
		kl_loss += self.gauss_kl(self.unsup_mu[label_source], self.unsup_logvar[label_source], \
									self.unsup_mu[unlabel_source], self.unsup_logvar[unlabel_source])
		kl_loss += self.gauss_kl(self.unsup_mu[unlabel_source], self.unsup_logvar[unlabel_source], \
									self.unsup_mu[label_source], self.unsup_logvar[label_source])
		# final loss
		auto_w = self.config.auto_weight
		kl_w = self.get_kl_weight()
		rec_w = self.config.rec_weight
		rl_w = self.config.rl_weight
		loss = auto_w*auto_loss + rl_w*rl_loss + rec_w*rec_loss + kl_w*kl_loss
		return {'auto': auto_loss, 'rec': rec_loss, 'rl': rl_loss, 'kl': kl_loss, 'exp_r': baseline}, loss


	def get_reward_from_rec(self, label_source, batch):
		batch_size, T = batch['word_idx'][label_source].size()
		beam_size = self.config.beam_size

		# reward from reconstruction
		assert self.output2[label_source]['trans']['mode'] == 'teacher_force'
		logits = self.output2[label_source]['trans']['logits'] # (B'=B*beam_size, T, V)
		index = batch['word_idx'][label_source].unsqueeze(1).repeat(1, beam_size, 1).view(batch_size*beam_size, T)
		value = torch.gather(torch.softmax(logits.detach(), dim=2), dim=2, index=index.unsqueeze(2)) # (B', T, 1)
		logprobs = torch.log(value.squeeze(2))#.detach() # (B', T) # use detach here?
		reward = torch.zeros(batch_size*beam_size).float().cuda()
		for new_batch_idx in range(batch_size*beam_size):
			old_batch_idx = new_batch_idx // beam_size
			s_len = batch['sent_len'][label_source][old_batch_idx]
			reward[new_batch_idx] = torch.mean(logprobs[new_batch_idx][:s_len])
		return reward.view(batch_size, beam_size) # (B, beam_size)
    '''