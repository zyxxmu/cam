# CaM: Cache Merging for Memory-efficient LLMs Inference ([Paper Link](https://openreview.net/forum?id=LCTmppB165))

Official PyTorch implementation of **CaM** (Cache Merging), as presented in our paper accepted at ICML 2024:

**CaM: Cache Merging for Memory-efficient LLMs Inference** </br>
Yuxin Zhang\*, Yuxuan Du\*, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, Rongrong Ji (\* indicates equal contribution) <br>

## Setup

Installation instructions can be found in [INSTALL.md](INSTALL.md).
## Run
**1. running on QA tasks**

Step1: generate the data for tasks
```bash
task=openbookqa #mathqa,boolq,copa,winogrande...
shots=0 
python -u generate_task_data.py \
  --output-file ${task}-${shots}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots}
```
Step2:  (Full cache/ Dense) generate the output with full cache
```bash
GPU=0
model=huggyllama/llama-7b
model_arch=llama   # llama / gpt-neox / opt
CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}
```
Step2: generate the output with  Cam（Zhang, et al. 2024）,StreamingLLM（Xiao, et al. 2023）, H2O（Zhang, et al. 2023）
```bash
GPU=4,7
method=cam #{streamingllm, cam, h2o}
model=huggyllama/llama-7b
model_arch=llama   # llama / gpt_neox / opt 
task=openbookqa
shots=0
CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}-${method}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --start_ratio 0.1 \
  --recent_ratio 0.1 \
  --enable_small_cache \
  --method ${method}
```
Step3: evaluate the result
```bash
task=openbookqa #mathqa,boolq,copa,winogrande...
method=cam #{streamingllm, cam, h2o}
shots=0
model_arch=llama   # llama / gpt_neox / opt 
python -u evaluate_task_result.py \
  --result-file ${task}-${shots}-${model_arch}-${method}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch} \
  --start-ratio 0.1 \
  --recent-ratio 0.1 \
  --ret-path ${task}-${shots}-${model_arch}-${method}.txt
```

**2. running on summary tasks(XSUM, CNN/Daily Mail)**
Step1：generate data for tasks (XSUM, CNN/Daily Mail)
```bash
python get_data.py \
  --dataset cnn_dailymail
```
Step2: generate the output with  Cam（Zhang, et al. 2024）, StreamingLLM（Xiao, et al. 2023）, H2O（Zhang, et al. 2023）
```bash
GPU=0,1,2,3
method=h2o #{streamingllm, cam, h2o}
task=cnndm #{xsum,multi_news,cnndm}
shot=0
ratio=0.2
model=huggyllama/llama-7b
model_arch=gpt_neox   # llama / gpt_neox / opt 
CUDA_VISIBLE_DEVICES=${GPU} python -u run_helm.py \
  --input_path data/${task}_${shot}shot.jsonl \
  --output_path ${task}-${shots}-${model_arch}-${method}.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
  --enable_small_cache \
  --start_ratio ${ratio} \
  --recent_ratio ${ratio} \
  --method ${method}
```
Step3: Evaluate
```bash
method=streamingllm #{streamingllm, cam, h2o}
task=cnndm #{xsum,multi_news,cnndm}
model_arch=llama
shots=0
python eval_helm.py \
  --task ${task} \
  --method ${method} \
  --model ${model_arch} \
  --input_path ${task}-${shots}-${model_arch}-${method}.jsonl\
  --output_path ${task}-${shots}-${model_arch}-${method}-rouge.txt
```

**3. running on generation tasks(wikitext, pg19 ...)**
Step1: switch to the directory for generation tasks
```bash
cd /cam/generate_task/examples
```
Step2: generate the output with Cam（Zhang, et al. 2024）, StreamingLLM（Xiao, et al. 2023）
```bash
method=streamingllm   # {cam, streamingllm}
model_name=huggyllama/llama-7b
task=wikitext # {wikitext, pg19}
python  examples/eval_long_ppl.py \
  --enable_start_recent_kv_cache \
  --enable_pos_shift \
  --model_name_or_path ${model_name} \
  --dataset_name ${task} \
  --start_size  32 \
  --recent_size  32 \
  --num_samples  10 \
  --method ${method} \
  --output_dir ${method}-${cache}-ppl.txt
```
（Note: If you cannot import name 'repeat_kv' from transformers library,  try install transformers library on version 4.33.0 by "pip install transformers==4.33.0"）

Step3: generate output with the full cache
```bash
model_name=huggyllama/llama-7b
task=wikitext # {wikitext, pg19}
python  examples/eval_long_ppl.py \
  --model_name_or_path ${model_name} \
  --dataset_name ${task} \
  --num_samples  100 \
  --output_dir dense-ppl.txt
```
The average perplexity(ppl) for selected samples is recorded at "output_dir "



## Citation

if you find this repo is helpful, please cite: 

```bibtex
@inproceedings{zhang2024CaM,
  title={CaM: Cache Merging for Memory-efficient LLMs Inference}, 
  author={Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, Rongrong Ji},
  year={2024},
  booktitle={International Conference on Machine Learning},
}
```



