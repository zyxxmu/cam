import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.opt.modeling_opt import OPTAttention


__all__ = ['convert_kvcache_opt_cam', 'OPTAttention_Mask_cam']

def local_cam_mask(value_states,attn_score,start_budget,recent_budget):
    value_states=value_states.unsqueeze(dim=0)
    merge_budget = 32
    attn_score=attn_score.unsqueeze(dim=0)
    #(1,nhead,k-token-1)
    token_index=attn_score.shape[-1]
    mean_attn=torch.mean(torch.cat((attn_score[0,:,:start_budget],attn_score[0,:,token_index-recent_budget:token_index]),dim=-1),dim=-1)
    merge_prob=attn_score[0,:,token_index-recent_budget]/mean_attn
    # merge_mask = torch.bernoulli(merge_prob.clamp(min=0,max=1))
    merge_mask = merge_prob > 1
    score1=value_states[:,:,token_index-recent_budget,...].clone()*merge_mask.unsqueeze(-1)/merge_budget
    value_states[:,:,token_index-recent_budget+1:token_index-recent_budget+merge_budget+1,:]+=score1.unsqueeze(2)
    value_states=value_states.squeeze(dim=0)
    return value_states

def local_full(value_states,attn_weights,start_budget,recent_budget):
    attn_weights=attn_weights.unsqueeze(dim=0)
    value_states=value_states.unsqueeze(dim=0)
    attn_score = attn_weights.sum(0).mean(1)
    merge_budget=64
    attn_score_ratio = attn_weights[:, :, attn_weights.shape[-1]-merge_budget:attn_weights.shape[-1], :].sum(0).mean(1)
    attn_score[:, :start_budget]=0

    selected_set = attn_score[:, :-recent_budget]
    _, keep_topk = selected_set.topk(k=16, dim=-1, largest=True)
    merge_mask = torch.zeros(attn_score.shape[0], attn_score.shape[1]).to(value_states.device)
    merge_mask = merge_mask.scatter(-1, keep_topk, 1)
    merge_mask = attn_score_ratio * merge_mask 
    #merge_mask = merge_ratio * merge_mask
    score1=torch.sum(value_states.clone()*merge_mask.unsqueeze(0).unsqueeze(-1),-2)
    value_states[:,:,start_budget,:]=score1
    value_states=value_states.squeeze(dim=0)
    return value_states

class OPTAttention_Mask_cam(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        start_ratio: float,
        recent_ratio: float,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.start_budget_ratio = start_ratio
        self.recent_budget_ratio = recent_ratio
        self.attention_masks_next = None 
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    def _reset_masks(self):
        self.attention_masks_next = None 
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
            value_states = local_cam_mask(value_states,self.previous_scores,self.start_budget,self.recent_budget)
            value_states[:,self.start_budget,:] *= 0.99
            past_key_value=(key_states.view(bsz,self.num_heads,-1,self.head_dim),value_states.view(bsz,self.num_heads,-1,self.head_dim)) if self.is_decoder else None
            
        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # attn_weights (heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
        current_scores_sum = attn_weights.sum(1) # (heads, k-tokens)

        # Accumulate attention scores
        if not self.previous_scores == None:
            current_scores_sum[:, :-1] += self.previous_scores #(Enlarge Sequence)
            attn_weights[:,:,self.start_budget]=1
        else:
            self.start_budget = int(self.start_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.start_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])
            
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        self.previous_scores = current_scores_sum #(heads, k-tokens)
        attn_mask = torch.zeros(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
        if attn_tokens_all > self.cache_budget:
            # activate most recent k-cache
            if not self.recent_budget == 0:
                attn_mask[:, attn_weights.shape[-1]-self.recent_budget:attn_weights.shape[-1]+1] = 1
            if not self.start_budget == 0:
                attn_mask[:,:self.start_budget+1]=1

        self.attention_masks_next = attn_mask.unsqueeze(1)

        if self.attention_masks_next is None:
            value_states = local_full(value_states,attn_weights,self.start_budget,self.recent_budget)
            past_key_value=(key_states.view(bsz,self.num_heads,-1,self.head_dim),value_states.view(bsz,self.num_heads,-1,self.head_dim)) if self.is_decoder else None

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def convert_kvcache_opt_cam(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_opt_cam(module, config)

        if isinstance(module, OPTAttention):
            model._modules[name] = OPTAttention_Mask_cam(
                embed_dim=module.embed_dim,
                num_heads=config.num_attention_heads,
                start_ratio = config.start_ratio,
                recent_ratio = config.recent_ratio,
                dropout=config.attention_dropout,
                is_decoder=True,
                bias=config.enable_bias,
            )
    return model

