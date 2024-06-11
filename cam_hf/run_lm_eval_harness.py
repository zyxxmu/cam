import argparse
import json, tqdm
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer

from utils_lm_eval.modify_llama_cam import convert_kvcache_llama_cam
from utils_lm_eval.modify_llama_sink import convert_kvcache_llama_sink
from utils_lm_eval.modify_llama_h2o import convert_kvcache_llama_h2o
from utils_lm_eval.modify_gptneox_cam import convert_kvcache_gpt_neox_cam, GPTNeoXAttention_Mask
from utils_lm_eval.modify_gptneox_sink import convert_kvcache_gpt_neox_sink, GPTNeoXAttention_Mask
from utils_lm_eval.modify_gptneox_h2o import convert_kvcache_gpt_neox_h2o, GPTNeoXAttention_Mask
from utils_lm_eval.modify_opt_h2o import convert_kvcache_opt_h2o, OPTAttention_Mask


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": { "cam": convert_kvcache_llama_cam, "streamingllm": convert_kvcache_llama_sink, "h2o": convert_kvcache_llama_h2o },
    "opt": {"h2o":convert_kvcache_opt_h2o,},
    "gpt_neox": {"cam":convert_kvcache_gpt_neox_cam,"streamingllm":convert_kvcache_gpt_neox_sink,"h2o":convert_kvcache_gpt_neox_h2o },
}

if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument("--cache-dir", type=str, default='../../checkpoint/')
    parser.add_argument("--method",type=str, default='cam')
    parser.add_argument("--start_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--merge_token", action='store_true')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    kwargs = { "torch_dtype": torch.float16, "device_map": "auto" }
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir, **kwargs)
    #model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
    print(args.model_type)
    if args.enable_small_cache:
        print('Enable Small Cache Size')
        print("The Compression Method is :",args.method)
        config.start_ratio = args.start_ratio
        config.recent_ratio = args.recent_ratio
        config.merge_token = args.merge_token
        checkpoint = copy.deepcopy(model.state_dict())
        
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type][args.method](model, config)
        model.load_state_dict(checkpoint)

    #model.half().eval()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            logits = model(input_ids).logits.log_softmax(dim=-1)
            
            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:] # skip first
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
            
            result['result'] = {
                "choices": [
                    {
                        "text": prompt, 
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')