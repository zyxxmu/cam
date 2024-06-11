import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from streaming_llm.kv_cache_sink import StartRecentKVCache_sink
from streaming_llm.kv_cache_cam import StartRecentKVCache_cam
from streaming_llm.utils import parse_args, load
from streaming_llm.pos_shift.modify_llama_sink import enable_llama_pos_shift_attention_sink
from streaming_llm.pos_shift.modify_llama_cam import enable_llama_pos_shift_attention_cam

enable_llama_pos_shift_attention={"cam":enable_llama_pos_shift_attention_cam, "streamingllm":enable_llama_pos_shift_attention_sink}
StartRecentKVCache={"cam":StartRecentKVCache_cam, "streamingllm":StartRecentKVCache_sink}

device = "cuda"
args = parse_args()

data = load_dataset(args.dataset_name,args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)
nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache== False and args.enable_pos_shift==False:
    print("the copression method is : dense")
else:
    print("the copression method is : ",args.method)

if args.enable_start_recent_kv_cache and args.method=="stramingllm":
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache[args.method](
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        enable_llama_pos_shift_attention[args.method](model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )
        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )
        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")
    
num_eval_tokens = 0
for text in data["text"][:args.num_samples]:
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))
    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)

            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
        
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break
ppl = torch.exp(torch.stack(nlls).mean())

with open(args.output_dir, "w") as f:
    f.write(f"model name :{args.model_name_or_path}\n")
    f.write(f"the compression method is :",args.method)
    f.write(f"mean ppl :{ppl.item()}\n")
    f.write(f"****************************************************\n")
print(f"mean ppl {ppl}")