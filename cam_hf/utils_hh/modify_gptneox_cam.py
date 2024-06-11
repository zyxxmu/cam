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

from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding, GPTNeoXAttention, apply_rotary_pos_emb


__all__ = ['convert_kvcache_gpt_neox_cam', 'GPTNeoXAttention_Mask_cam']

def local_cam_mask(value_states,attn_score,start_budget,recent_budget):
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
    return value_states
def local_full(value_states,attn_weights,start_budget,recent_budget):
    attn_score = attn_weights.sum(0).mean(1)
    merge_budget=128
    attn_score_ratio = attn_weights[:, :, attn_weights.shape[-1]-merge_budget:attn_weights.shape[-1], :].sum(0).mean(1)
    #merge_ratio = attn_weights[:, :, attn_weights.shape[-1]-512:attn_weights.shape[-1], :].sum(0).sum(1)/512
    attn_score[:, :start_budget]=0
    selected_set = attn_score[:, :-recent_budget]
    _, keep_topk = selected_set.topk(k=16, dim=-1, largest=True)
    merge_mask = torch.zeros(attn_score.shape[0], attn_score.shape[1]).to(value_states.device)
    merge_mask = merge_mask.scatter(-1, keep_topk, 1)
    merge_mask = attn_score_ratio * merge_mask 
    #merge_mask = merge_ratio * merge_mask
    score1=torch.sum(value_states.clone()*merge_mask.unsqueeze(0).unsqueeze(-1),-2)
    value_states[:,:,start_budget,:]=score1
    return value_states

class GPTNeoXAttention_Mask_cam(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.start_budget_ratio = config.start_ratio
        self.recent_budget_ratio = config.recent_ratio
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

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights, present = self._attn(query, key, value, use_cache,attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, use_cache, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        past_key_value=0
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        if self.attention_masks_next is not None:
            attn_scores = attn_scores * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_scores.dtype).min
            value = local_cam_mask(value,self.previous_scores,self.start_budget,self.recent_budget)
            value[:,:,self.start_budget,:] *= 0.99
            key = key.view(batch_size, num_attention_heads, key_length, attn_head_size)
            past_key_value=(key,value) if use_cache else None
    
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)


        # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
        # offset = attn_weights.gt(0).sum(0).sum(1)

        # Accumulate attention scores
        if not self.previous_scores == None:
            current_scores_sum[:, :-1] += self.previous_scores #(Enlarge Sequence)
            attn_weights[:,:,:,self.start_budget] = 1
        else:
            self.start_budget = int(self.start_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.start_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])

        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        self.previous_scores = current_scores_sum #(heads, k-tokens)
        attn_mask = torch.zeros(attn_weights.shape[1], attn_weights.shape[-1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        if attn_weights.shape[-1] > self.cache_budget:
            if not self.recent_budget == 0:
                attn_mask[:,attn_weights.shape[-1]-self.recent_budget:attn_weights.shape[-1]+1] = 1
            if not self.start_budget ==0 :
                attn_mask[:,:self.start_budget+1] = 1
        
        if self.attention_masks_next == None:
            value = local_full(value,attn_weights,self.start_budget,self.recent_budget)
            key = key.view(batch_size, num_attention_heads, key_length, attn_head_size)
            past_key_value=(key,value) if use_cache else None

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)
        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights, past_key_value

def convert_kvcache_gpt_neox_cam(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_gpt_neox_cam(module, config)

        if isinstance(module, GPTNeoXAttention):
            model._modules[name] = GPTNeoXAttention_Mask_cam(config)

            target_device = next(module.parameters()).device
            for param in model._modules[name].parameters():
                param.data = param.data.to(target_device)
            for buffer in model._modules[name].buffers():
                buffer.data = buffer.data.to(target_device)
            model._modules[name].half() 

    return model


