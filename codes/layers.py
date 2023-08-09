from logging import log
from re import S
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings
from transformers.modeling_utils import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
    prune_linear_layer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import GPT2Config, load_tf_weights_in_gpt2
from modeling_utils import PreTrainedModel

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple
import math

from transformers.utils import logging
logger = logging.get_logger(__name__)

is_amp_available = True
from torch.cuda.amp import autocast

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

GPT2_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPT2Model):
            module.gradient_checkpointing = value


#---------------------------------------------------------------
@dataclass
class GPT2VAEOutputWithPast(ModelOutput):
    gen_loss: Optional[torch.FloatTensor] = None
    bow_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    infonce_loss: Optional[torch.FloatTensor] = None
    gen_logits: torch.FloatTensor = None
    cl_logits: torch.FloatTensor = None
    gen_kl: Optional[torch.FloatTensor] = None
    cl_kl: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class GPT2VAEPretrainOutputWith(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


# -----------------------------------------------------------
class LossWrapper(object):
    def __init__(self, pad_idx, kl_lambda=0, label_smoothing_factor=0.0):

        self._criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self._cl_criterion = nn.CrossEntropyLoss()
        self._criterion2 = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')
        self._klcriterion = nn.KLDivLoss(reduction="batchmean")
        
        if label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
        
        self.pad_idx = pad_idx
        self.kl_lambda = kl_lambda
    
    
    def gaussianKL(self, recog_mu, recog_log_sigma_sq, prior_mu, prior_log_sigma_sq):
        '''
        KLD = -0.5 * torch.sum(
            1 + (recog_log_sigma_sq - prior_log_sigma_sq)
            - torch.div( torch.pow(prior_mu - recog_mu, 2),  prior_log_sigma_sq.exp())
            - torch.div( recog_log_sigma_sq.exp(), prior_log_sigma_sq.exp() )
        )
        '''
        kld = -0.5*(
            1 + (recog_log_sigma_sq - prior_log_sigma_sq)
            - torch.div( torch.pow(prior_mu - recog_mu, 2),  prior_log_sigma_sq.exp())
            - torch.div( recog_log_sigma_sq.exp(), prior_log_sigma_sq.exp() )
        )
        
        if self.kl_lambda > 0:
            kld = torch.maximum(kld, torch.ones_like(kld)*self.kl_lambda)
        
        return kld.sum(dim=-1).mean()


    def genCELoss(self, logits, tgts, attn_mask, type_ids, pretrain=False):
        # [CLS] A [SEP] B [SEP]
        # logits (B, L, V), tgts (B, L), type_ids (B, L)
        
        #mask = (1-attn_mask).bool() & (1-type_ids).bool()
        #masked_labels = tgts.masked_fill(mask, self.pad_idx)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgts[..., 1:].contiguous()
        shift_type_ids = type_ids[..., 1:].contiguous()

        if not pretrain:
            masked_labels = shift_labels.masked_fill(shift_type_ids==0, self.pad_idx)
        else:
            masked_labels = shift_labels
        
        if self.label_smoother is not None:
            loss = self.label_smoother([logits], masked_labels)
        else:
            # Flatten the tokens
            flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
            flatten_labels = masked_labels.view(-1)
            loss = self._criterion(flatten_logits, flatten_labels)
        
        return loss
    
    def genCELoss2(self, logits, tgts, attn_mask, type_ids, pretrain=False):
        # [CLS] A [SEP] B [SEP]
        # logits (B, L, V), tgts (B, L), type_ids (B, L)
        
        #mask = (1-attn_mask).bool() & (1-type_ids).bool()
        #masked_labels = tgts.masked_fill(mask, self.pad_idx)
        batch_size = logits.size(0)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgts[..., 1:].contiguous()
        shift_type_ids = type_ids[..., 1:].contiguous()

        if not pretrain:
            masked_labels = shift_labels.masked_fill(shift_type_ids==0, self.pad_idx)
        else:
            masked_labels = shift_labels
        
        if self.label_smoother is not None:
            loss = self.label_smoother([logits], masked_labels)
        else:
            # Flatten the tokens
            flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
            flatten_labels = masked_labels.view(-1)
            loss = self._criterion2(flatten_logits, flatten_labels)
            loss = loss.view(batch_size, -1).sum(-1)
        return loss
    
    def stKL(self, prob, prob_t, attn_mask, type_ids):
        #batch_size = logits.size(0)
        #prob=F.softmax(logits,dim=-1)
        #prob_t=F.softmax(target_logit,dim=-1)
        
        shift_logits = prob[..., :-1, :].contiguous()
        shift_labels = prob_t[..., 1:].contiguous()
        shift_type_ids = type_ids[..., 1:].contiguous()
        
        shift_type_ids = shift_type_ids.unsqueeze(-1).tile(shift_logits.shape[-1])
        #mask!
        m=1/shift_logits.shape[-1]
        shift_logits=shift_logits.masked_fill(shift_type_ids==0,m)
        shift_labels=shift_logits.masked_fill(shift_type_ids==0,m)
        
        flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
        flatten_labels = shift_labels.view(-1, shift_labels.size(-1))
        loss = self._klcriterion(flatten_logits, flatten_labels)
        # = loss.view(batch_size, -1).sum(-1)
        
        #sum t*log(t/l)
        return loss
    
    
    def clCELoss(self, logits, labels):
        return self._cl_criterion(logits, labels)

    def bow_loss(self, logits, tgts, mask, type_ids):
        # logits: (B, V)
        # tgts: (B, L)
        
        return self.genCELoss(logits.unsqueeze(1).repeat(1, tgts.size(1), 1),
            tgts, mask, type_ids)

# ---------------------------------------------
class MLP(nn.Module):
    def __init__(self, ori_input_size, layer_sizes, activs=None,
        drop_ratio=0.0, no_drop=False):
        super(MLP, self).__init__()

        layer_num = len(layer_sizes)

        orderedDic = OrderedDict()
        input_size = ori_input_size
        for i, (layer_size, activ) in enumerate(zip(layer_sizes, activs)):
            linear_name = 'linear_' + str(i)
            orderedDic[linear_name] = nn.Linear(input_size, layer_size)
            input_size = layer_size

            if activ is not None:
                assert activ in ['tanh', 'relu', 'leakyrelu']

            active_name = 'activ_' + str(i)
            if activ == 'tanh':
                orderedDic[active_name] = nn.Tanh()
            elif activ == 'relu':
                orderedDic[active_name] = nn.ReLU()
            elif activ == 'leakyrelu':
                orderedDic[active_name] = nn.LeakyReLU(0.1)


            if (drop_ratio > 0) and (i < layer_num-1) and (not no_drop):
                orderedDic["drop_" + str(i)] = nn.Dropout(drop_ratio)

        self.mlp = nn.Sequential(orderedDic)


    def forward(self, inps):
        return self.mlp(inps)



class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.transform_act_fn = ACT2FN['gelu']
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states



class GPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        
        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.c_key = nn.Linear(config.hidden_size, config.hidden_size)
        self.c_value = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.c_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.c_dropout = nn.Dropout(config.hidden_pdrop)

        self.pruned_heads = set()

        self.z_inject = nn.Linear(config.n_embd+config.n_latent, config.n_embd)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        self.c_query = prune_linear_layer(self.c_query, index)
        self.c_key = prune_linear_layer(self.c_key, index)
        self.c_value = prune_linear_layer(self.c_value, index)
        
        self.c_proj = prune_linear_layer(self.c_proj, index, dim=1)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, with_causal_mask=True):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
            
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if only "normal" attention layer implements causal mask
        if with_causal_mask:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
                attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
                attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
        with_causal_mask=True,
        latent_variable=None,
    ):  
        
        query = self.c_query(hidden_states)
        key = self.c_key(hidden_states)
        value = self.c_value(hidden_states)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, with_causal_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # (B, L, H)
        

        if latent_variable is not None:
            attn_output = self.z_inject(torch.cat([attn_output,
                latent_variable.unsqueeze(dim=1).repeat(1, attn_output.size(1), 1)], dim=-1))

        attn_output = self.c_proj(attn_output)
        attn_output = self.c_dropout(attn_output)
        attn_output = self.c_ln(attn_output + hidden_states)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)



#----------------------------------------------
class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        
        self.intermediate_dense = nn.Linear(config.n_embd, inner_dim)
        self.intermediate_act_fn = ACT2FN['gelu']

        self.out_dense = nn.Linear(inner_dim, config.n_embd)
        self.out_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.out_dropout = nn.Dropout(config.hidden_pdrop)
        

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
        with_causal_mask=True,
        latent_variable=None,
    ):
        # no ln for inps for bert
        # residual = hidden_states
        #hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            with_causal_mask=with_causal_mask,
            latent_variable=latent_variable,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # ----------------------------
        intermediate_output = self.intermediate_act_fn(self.intermediate_dense(attn_output))
        intermediate_output = self.out_dropout(self.out_dense(intermediate_output))
        layer_output = self.out_ln(intermediate_output + attn_output)

        '''
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        '''

        if use_cache:
            outputs = (layer_output,) + outputs
        else:
            outputs = (layer_output,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
#-----------------------------------------------------------------------------------------
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.wie = nn.Embedding(6, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        #self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        self.wie = self.wie.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        self.wie = self.wie.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        with_causal_mask=True,
        latent_variable=None,
        input_embeddings=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
            device = input_ids.device
            inputs_embeds = self.wte(input_ids)
        elif input_embeddings is not None:
            input_shape = attention_mask.size()
            batch_size = input_embeddings.shape[0]
            device = input_embeddings.device
            inputs_embeds=input_embeddings
        else:
            raise NotImplementedError()
        
        
            
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        '''
        print("pos id in transformer")
        print("start-------------------------")
        print(position_ids[0, :])
        print("end---------------------------")
        '''
        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        encoder_attention_mask = None
        
        #inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        # embedding dropout
        # ln and drop applied to embeeding instead of output compared to GPT-2
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    with_causal_mask=with_causal_mask,
                    latent_variable=latent_variable,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    with_causal_mask=with_causal_mask,
                    latent_variable=latent_variable,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # no ln applied to output for BERT
        #hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class NoiseLayer(nn.Module):
    """Add noise to words,
    wrapper class of noise function from FAIR (upon some modification):
    https://github.com/facebookresearch/UnsupervisedMT/blob/master/NMT/src/trainer.py
    """
    def __init__(self, word_blank, word_dropout, word_shuffle,
                 pad_index, blank_index, eos_index, bpe_encode=False):
        """
        Args:
            word_blank (float): blank out probability, 0 to disable
            word_dropout (float): drop out probability, 0 to disable
            word_shuffle (float): should be larger than 1., 0 to disable,
                                  larger value means more shuffling noise
            pad_index (int): the pad index
            blank_index (int): the index used to blank out separate words
        """
        super(NoiseLayer, self).__init__()
        self.blank_prob = word_blank
        self.dropout_prob = word_dropout
        self.shuffle_weight = word_shuffle

        self.pad_index = pad_index
        self.blank_index = blank_index
        self.eos_index = eos_index

    def noising(self, words, lengths):
        """perform shuffle, dropout, and blank operations,
        note that the input is required to have bos_index at the start and
        eos_index at the end
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        if self.blank_prob == 0:
            return x, l
        assert 0 < self.blank_prob < 1

        # define words to blank
        # bos_index = self.bos_index[lang_id]
        # assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.blank_prob
        keep[0] = 1  # do not blank the start sentence symbol

        # # be sure to blank entire words
        # bpe_end = self.bpe_end[lang_id][x]
        # word_idx = bpe_end[::-1].cumsum(0)[::-1]
        # word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(len(l)):
            # assert x[l[i] - 1, i] == eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.blank_index for j, w in enumerate(words)]
            new_s.append(self.eos_index)

            sentences.append(new_s)
        # re-construct input
        x2 = x.new_full((max(l), len(l)), fill_value=self.pad_index)
        for i in range(len(l)):
            x2[:l[i], i].copy_(x.new_tensor(sentences[i]))
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        if self.dropout_prob == 0:
            return x, l
        assert 0 < self.dropout_prob < 1

        # define words to drop
        # bos_index = self.bos_index[lang_id]
        # assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.dropout_prob
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        # bpe_end = self.bpe_end[lang_id][x]
        # word_idx = bpe_end[::-1].cumsum(0)[::-1]
        # word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(len(l)):
            assert x[l[i] - 1, i] == self.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.eos_index)

            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = lengths
        x2 = x.new_full((max(l2), len(l2)), fill_value=self.pad_index)
        for i in range(len(l2)):
            x2[:l2[i], i].copy_(x.new_tensor(sentences[i]))
        return x2, l2

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        if self.shuffle_weight == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.shuffle_weight, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        # bpe_end = self.bpe_end[lang_id][x]
        # word_idx = bpe_end[::-1].cumsum(0)[::-1]
        # word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.shuffle_weight > 1
        x2 = x.clone()
        for i in range(len(l)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            # scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l
