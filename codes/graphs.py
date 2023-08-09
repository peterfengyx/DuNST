# coding=utf-8
from numpy import empty
import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import CausalLMOutputWithPast
import math

from layers import (
    GPT2PreTrainedModel,
    GPT2Model,
    MLP,
    LMHead,
    LossWrapper,
    GPT2VAEOutputWithPast
)

from config import extra_args
from utils import log_sum_exp

class GPT2VAE(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = LMHead(config)
    
        self.label_embed = nn.Embedding(config.n_label, config.n_label_embd)
        
        # gen posteriori, p(z|x,y,c)
        self.posteriori = MLP(config.n_embd + config.n_label_embd,
            layer_sizes=[config.n_embd, config.n_embd//2, config.n_latent*2],
            activs=['leakyrelu', 'leakyrelu', None],
            drop_ratio=config.hidden_pdrop)
        
        # gen prior, p(z|y,c)
        self.gen_prior = MLP(config.n_label_embd,
            layer_sizes=[config.n_label_embd, config.n_label_embd//2, config.n_latent*2],
            activs=['leakyrelu', 'leakyrelu', None],
            drop_ratio=config.hidden_pdrop)

        # cl prior, p(z|x,c)
        self.cl_prior = MLP(config.n_embd,
            layer_sizes=[config.n_embd, config.n_embd//2, config.n_latent*2],
            activs=['leakyrelu', 'leakyrelu', None],
            drop_ratio=config.hidden_pdrop)
        
        self.cl_layer = nn.Sequential(
            nn.Linear(config.n_latent, 64), nn.Tanh(),
            nn.Dropout(config.cl_pdrop),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Dropout(config.cl_pdrop),
            nn.Linear(32, 2),
        )
        
        self.bow_layer = nn.Sequential(
            nn.Linear(config.n_latent, config.n_embd), nn.Tanh(),
            nn.Dropout(config.hidden_pdrop)
        )

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.config = config
        
        self.loss_fct = LossWrapper(config.pad_token_id, config.kl_lambda)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_seq_rep(self, inp_ids, attn_mask, type_ids,input_embeddings=None):
        
        outputs = self.transformer(
            inp_ids,
            attention_mask=attn_mask,
            token_type_ids=type_ids,
            output_hidden_states=False,
            return_dict=True,
            with_causal_mask=False,
            input_embeddings=input_embeddings
        )
        return outputs.last_hidden_state[:, 0, :] # (B, H), cls as feature


    def get_gen_outputs(self, seq_ids, seq_attn_mask,
        seq_type_ids, latent_variable,input_embeddings=None):
        
        # q(x|z,y,c)
        hidden_states = self.transformer(
            input_ids=seq_ids,
            attention_mask=seq_attn_mask,
            token_type_ids=seq_type_ids,
            return_dict=True,
            with_causal_mask=True,
            latent_variable=latent_variable,
            input_embeddings=input_embeddings
        ).last_hidden_state

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.device)

        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
    
    def logit2prob(self,logit,temp=1.0):
        prob=F.softmax(logit/temp,dim=-1)
        return prob
    
    #def get_logit(self.input_dic):
        
    
    def sample_z(self, rep, z_type, task_type):
        # z_type: prior or post
        # task_type: gen or cl
        assert z_type in ['prior', 'post']
        assert task_type in ['gen', 'cl']

        if z_type == 'post':
            latent = self.posteriori(rep)
        elif z_type == 'prior' and task_type == 'gen':
            latent = self.gen_prior(rep)
        elif z_type == 'prior' and task_type == 'cl':
            latent = self.cl_prior(rep)

        (mu, log_sigma) = latent.split(self.config.n_latent, dim=-1)
        z = mu + log_sigma.exp() * torch.randn_like(mu)
        return z, mu, log_sigma

    def forward(self, input_dic, is_evaluate=False, n_sample=1): 
        
        if is_evaluate:
            return self.do_classification(input_dic, nsamples=n_sample)
        
        
        
        if 'past_logit' in input_dic:
            inp_prob=self.logit2prob(input_dic['past_logit'],temp=extra_args.gen_temperature)
            inp_emb=torch.matmul(inp_prob,self.transformer.wte.weight)
            xc_h = self.get_seq_rep(None,
                input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'], inp_emb)
            
            # get rep of (c)
            c_h = xc_h
            
            y_h = self.label_embed(input_dic['cl_labels'])
            
            # ------------------------------------------------------------------------------
            # generation direction
            # KL[p(z|x,y,c)||q(z|y,c)] - log q(x|y,z,c)
            # get p(z|x,c,y) for gen
            gen_z_post, gen_post_mu, gen_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "gen")
            # get q(z|y,c) for gen
            gen_z_prior, gen_prior_mu, gen_prior_logs = self.sample_z(y_h, "prior", "gen")

            gen_logits = self.get_gen_outputs(
                None, input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'], gen_z_post, inp_emb
            )
            
            gen_prob=F.softmax(gen_logits,dim=-1)
            
            gen_ce_loss = self.loss_fct.stKL(gen_prob,
                inp_prob, input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'])
            
            gen_kl_loss = self.loss_fct.gaussianKL(gen_post_mu, gen_post_logs,
                gen_prior_mu, gen_prior_logs)
            

            # ------------------------------------------------------------------------------
            # classifier KL[p(z|x,y,c)||q(z|x,c)] - log q(y|x,z,c)
            # get p(z|x,c,y) for cl
            
            cl_z_post, cl_post_mu, cl_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "cl")
            cl_z_prior, cl_prior_mu, cl_prior_logs = self.sample_z(xc_h, "prior", "cl")        
            cl_logits = self.cl_layer(cl_z_post)
            
            cl_kl_loss = self.loss_fct.gaussianKL(cl_post_mu, cl_post_logs,
                cl_prior_mu, cl_prior_logs)  
            cl_ce_loss = self.loss_fct.clCELoss(cl_logits, input_dic['cl_labels'])

            # bow loss
            bow_logits = self.lm_head(self.bow_layer(gen_z_post))

            bow_loss = self.loss_fct.bow_loss(bow_logits, input_dic['seq_ids'],
                input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
            
            # ------------------------------------------------------------------------------
            return GPT2VAEOutputWithPast(
                gen_loss=gen_ce_loss, cl_loss=cl_ce_loss,
                gen_kl=gen_kl_loss, cl_kl=cl_kl_loss, bow_loss=bow_loss,
                gen_logits=gen_logits, cl_logits=cl_logits,
            ) 

        # get rep of (x,c)  
        xc_h = self.get_seq_rep(input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        
        # get rep of (c)
        if 'body_ids' in input_dic:
            c_h = self.get_seq_rep(input_dic['body_ids'], 
                                   input_dic['body_attn_mask'], 
                                   input_dic['body_token_type_ids'])
        else:
            c_h = xc_h
        
        y_h = self.label_embed(input_dic['cl_labels'])
        
        # ------------------------------------------------------------------------------
        # generation direction
        # KL[p(z|x,y,c)||q(z|y,c)] - log q(x|y,z,c)
        # get p(z|x,c,y) for gen
        gen_z_post, gen_post_mu, gen_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "gen")
        # get q(z|y,c) for gen
        gen_z_prior, gen_prior_mu, gen_prior_logs = self.sample_z(y_h, "prior", "gen")

        gen_logits = self.get_gen_outputs(
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'], gen_z_post
        )
        gen_ce_loss = self.loss_fct.genCELoss(gen_logits,
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'])
        
        gen_kl_loss = self.loss_fct.gaussianKL(gen_post_mu, gen_post_logs,
            gen_prior_mu, gen_prior_logs)
        
        # ------------------------------------------------------------------------------
        # classifier KL[p(z|x,y,c)||q(z|x,c)] - log q(y|x,z,c)
        # get p(z|x,c,y) for cl
        cl_z_post, cl_post_mu, cl_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "cl")
        cl_z_prior, cl_prior_mu, cl_prior_logs = self.sample_z(xc_h, "prior", "cl")        
        cl_logits = self.cl_layer(cl_z_post)
        
        cl_kl_loss = self.loss_fct.gaussianKL(cl_post_mu, cl_post_logs,
            cl_prior_mu, cl_prior_logs)  
        cl_ce_loss = self.loss_fct.clCELoss(cl_logits, input_dic['cl_labels'])

        # bow loss
        bow_logits = self.lm_head(self.bow_layer(gen_z_post))
        bow_loss = self.loss_fct.bow_loss(bow_logits, input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        
        # ------------------------------------------------------------------------------
        return GPT2VAEOutputWithPast(
            gen_loss=gen_ce_loss, cl_loss=cl_ce_loss,
            gen_kl=gen_kl_loss, cl_kl=cl_kl_loss, bow_loss=bow_loss,
            gen_logits=gen_logits, cl_logits=cl_logits,
        ) 
        
    def do_classification(self, input_dic, nsamples=1, with_sample=False, sample_n=None):
        # get rep 
        xc_h = self.get_seq_rep(input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        # get q(z|x,c) for cl
        _, cl_prior_mu, cl_prior_logs = self.sample_z(xc_h, "prior", "cl")
        
        y_h = self.label_embed(input_dic['cl_labels'])
        
        if nsamples==1:
            #gen_z_prior, gen_prior_mu, gen_prior_logs = self.sample_z(y_h, "prior", "gen")
            gen_z_post, gen_post_mu, gen_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "gen")
            
            gen_logits = self.get_gen_outputs(
                input_dic['seq_ids'], input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'], gen_z_post
            )
            gen_ce_loss = self.loss_fct.genCELoss(gen_logits,
                input_dic['seq_ids'], input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'])
        else:
            gen_ce_loss=self.nll_iw(input_dic, xc_h, y_h, nsamples,nz=1)
        cl_logits = self.cl_layer(cl_prior_mu)
        
        cl_ce_loss = self.loss_fct.clCELoss(cl_logits, input_dic['cl_labels'])
        return {'cl_loss': cl_ce_loss, 'cl_logits': cl_logits, 'gen_loss': gen_ce_loss}


    def do_generation(self, input_ids,
        token_type_ids, attn_mask, position_ids,
        past_key_values, latent_variable):
        
        # q(x|z,y,c)
        outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            return_dict=True, use_cache=True,
            with_causal_mask=True,
            past_key_values=past_key_values,
            latent_variable=latent_variable
        )

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = outputs.last_hidden_state.to(self.lm_head.weight.device)
        else:
            hidden_states = outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=outputs.past_key_values
        )
    
    def get_gen_prior_z(self, c_h, labels, n):
        y_h = self.label_embed(labels)

        # get q(z|y,c) for gen
        if c_h==None:
            latent = self.gen_prior(y_h)
        else:
            latent = self.gen_prior(torch.cat([c_h, y_h], dim=-1))
        mu, log_sigma = latent.split(self.config.n_latent, dim=-1)
        z = mu + log_sigma.exp() * torch.randn_like(mu)

        mu0 = mu.unsqueeze(1).repeat(1, n, 1)
        log_sigma0 = log_sigma.unsqueeze(1).repeat(1, n, 1)
        z0 = mu0 + log_sigma0.exp() * torch.randn_like(mu0)

        return z, z0
    
    def get_gen_post_z(self, xc_h, y_h, n):
        #y_h = self.label_embed(labels)

        # get q(z|y,c) for gen
        if xc_h==None:
            latent = self.posteriori(y_h)
        else:
            latent = self.posteriori(torch.cat([xc_h, y_h], dim=-1))
        mu, log_sigma = latent.split(self.config.n_latent, dim=-1)
        z = mu + log_sigma.exp() * torch.randn_like(mu)

        mu0 = mu.unsqueeze(1).repeat(1, n, 1)
        log_sigma0 = log_sigma.unsqueeze(1).repeat(1, n, 1)
        z0 = mu0 + log_sigma0.exp() * torch.randn_like(mu0)

        return z, z0, mu, log_sigma
    
    def eval_inference_dist(self, z, mu, logvar):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        #nz = z.size(2)
        #mu, logvar = param

        # (batch_size, 1, nz)
        #mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()
        
        post = torch.distributions.normal.Normal(mu, var)
        return post.log_prob(z).sum(dim=-1)
        
        '''
        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density
        '''
    
    def eval_prior_dist(self, zrange, y_h, c_h=None):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        if c_h==None:
            latent = self.gen_prior(y_h)
        else:
            latent = self.gen_prior(torch.cat([c_h, y_h], dim=-1))
        mu, log_sigma = latent.split(self.config.n_latent, dim=-1)
        scale=torch.exp(log_sigma)
        prior = torch.distributions.normal.Normal(mu, scale)
        return prior.log_prob(zrange).sum(dim=-1)
    
    def nll_iw(self, input_dic, xc_h, y_h, nsamples,nz=1):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x0, x1:  two different tokenization results of x, where x is the data tensor with shape (batch, *). 
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        #.      this problem is to be solved in order to speed up

        tmp = []
        
        for _ in range(int(nsamples//nz)):
            # [batch, ns, nz]

            # Chunyuan:
            # encoding into bert features
            z, z0, mu, log_sigma = self.get_gen_post_z(xc_h, y_h, nz)
            z=z0[:,0,:]
            #pooled_hidden_fea = self.encoder(x0)[1]

            # param is the parameters required to evaluate q(z|x)
            #z, param = self.encoder_sample(pooled_hidden_fea, ns)

            # [batch, ns]
            #gen_z_post, gen_post_mu, gen_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "gen")
            
            gen_logits = self.get_gen_outputs(
                input_dic['seq_ids'], input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'], z
            )
            gen_ce_loss = self.loss_fct.genCELoss(gen_logits,
                input_dic['seq_ids'], input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'])
            
            log_prior_ll = self.eval_prior_dist(z0, y_h)
            log_gen_ll = -gen_ce_loss
            log_infer_ll = self.eval_inference_dist(z0, mu, log_sigma)

            tmp.append(log_prior_ll+log_gen_ll - log_infer_ll)
            print("prior",log_prior_ll)
            print("gen:",log_gen_ll)
            print("infer:",log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return ll_iw
    
    def get_klloss(self,input_dic):
        xc_h = self.get_seq_rep(input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        
        # get rep of (c)
        if 'body_ids' in input_dic:
            c_h = self.get_seq_rep(input_dic['body_ids'], 
                                   input_dic['body_attn_mask'], 
                                   input_dic['body_token_type_ids'])
        else:
            c_h = xc_h
        
        y_h = self.label_embed(input_dic['cl_labels'])
        
        gen_z_post, gen_post_mu, gen_post_logs = self.sample_z(torch.cat([xc_h, y_h], dim=-1), "post", "gen")
        gen_z_prior, gen_prior_mu, gen_prior_logs = self.sample_z(y_h, "prior", "gen")
        
        gen_kl_loss = self.loss_fct.gaussianKL(gen_post_mu, gen_post_logs,
            gen_prior_mu, gen_prior_logs)
        return gen_kl_loss
    
    def iw_sample(self,input_dic):
        xc_h = self.get_seq_rep(input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        y_h = self.label_embed(input_dic['cl_labels'])
        z, z0, mu, log_sigma = self.get_gen_post_z(xc_h, y_h, 1)
        y_h = self.label_embed(input_dic['cl_labels'])
        log_prior = self.eval_prior_dist(z, y_h)
        log_infer = self.eval_inference_dist(z, mu, log_sigma)
        gen_logits = self.get_gen_outputs(
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'], z
        )
        gen_ce_loss = self.loss_fct.genCELoss2(gen_logits,
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'])
        #print(gen_ce_loss)
        log_gen = -gen_ce_loss
        log_likelihood = log_gen + log_prior - log_infer
        return log_gen, log_likelihood
