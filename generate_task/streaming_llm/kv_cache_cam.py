import torch

def min_max_scaler(tensor):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    tensor = (tensor - min_val) / (max_val - min_val) 
    return tensor


def slice2d(x, start, end, pos, attn_weights=None, merge=False):
    if pos=="recent" and merge==True and attn_weights != None:
        #merge_prob = (torch.mean(attn_weights[...,start-merge_num:start], dim=1)/torch.mean(attn_weights[...,start::]))
        merge_num = 1
        merge_prob = attn_weights[...,start-merge_num:start]/torch.mean(attn_weights[...,start::], dim=-1).unsqueeze(2)
        merge_mask = torch.bernoulli(merge_prob.clamp(min=0,max=1)).squeeze(2).unsqueeze(-1)
        x[:,:,start:start+4, ...]=x[:,:,start:start+4, ...]+(torch.sum(x[:,:,start-merge_num:start, ...]*merge_mask, dim=2)).unsqueeze(2)/4
    return x[:, :, start:end, ...]


def slicev(x, start, end, pos, accum_attn=None, attn_weights=None, merge=False):
    if accum_attn == None:
        accum_attn = attn_weights
    else:
        tmp = attn_weights[..., :accum_attn.shape[-1]] + accum_attn
        accum_attn = torch.cat([tmp, attn_weights[..., accum_attn.shape[-1]:]], dim=-1)
        #print(x.shape)
    

    if pos=="recent" and merge==True and attn_weights != None:
        merge_num = 16
        #merge_prob = accum_attn[..., start-1]/torch.max(accum_attn[..., start+merge_num:start+merge_num+merge_num], dim=-1)[0]
        merge_prob = accum_attn[..., start-1]/torch.mean(accum_attn[..., start:start+merge_num], dim=-1)
        #merge_prob = attn_weights[...,start-1]/torch.mean(attn_weights[...,start:start+merge_num], dim=-1)
        if torch.isnan(merge_prob).any():
            merge_prob[torch.isnan(merge_prob)] = 0
        merge_mask = torch.bernoulli(merge_prob.clamp(min=0,max=1))
        to_merge = x[:,:,start-1, ...]*merge_mask/merge_num
        # x[:,:,start+merge_num:start+merge_num+merge_num, ...] += to_merge.unsqueeze(2)
        x[:,:,start:start+merge_num, ...] += to_merge.unsqueeze(2)
        accum_attn = torch.cat([accum_attn[..., :start-1], accum_attn[..., start:]], dim=-1)
    return x[:, :, start:end, ...], accum_attn

def slice3d(x, start, end, pos, merge=False, merge_ratio=0):
    if pos=="recent" and merge==True:
        x[:,:,start, ...]=(x[:,:,start, ...]+x[:,:,start-1, ...]*merge_ratio)
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end, pos, merge=False, merge_ratio=0):
    if pos=="recent" and merge==True:
        x[:,:,start, ...]=(x[:,:,start, ...]+x[:,:,start-1, ...]*merge_ratio)
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
    4: slicev,
}

class StartRecentKVCache_cam:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice_merge = DIM_TO_SLICE[4]
        self.accum_attn = None

    def __call__(self,attn_weights, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        k=past_key_values[0]
        v=past_key_values[1]
        if seq_len <= self.cache_size:
            merge_v, self.accum_attn = self.v_slice_merge(v, seq_len - self.recent_size, seq_len, pos="recent", accum_attn=self.accum_attn, attn_weights=attn_weights, merge=False)
            return past_key_values
        
        
        # print(self.accum_attn)
        merge_v, self.accum_attn = self.v_slice_merge(v, seq_len - self.recent_size, seq_len, pos="recent", accum_attn=self.accum_attn, attn_weights=attn_weights, merge=True)
        #import pdb; pdb.set_trace()
        #self.accum_attn = 
        return [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size, pos="start"),
                        self.k_slice(k, seq_len - self.recent_size, seq_len, pos="recent", attn_weights=attn_weights, merge=False),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size, pos="start"),
                        merge_v
                    ],
                    dim=self.v_seq_dim,
                ),
            ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size, pos="start"),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len, pos="recent", merge=False
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size, pos="start"),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len, pos="recent", merge=True
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start, pos="start"),
                        self.k_slice(k, end, seq_len, pos="recent",merge=False),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start, pos="start"),
                        self.v_slice(v, end, seq_len, pos="recent",merge=True),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
