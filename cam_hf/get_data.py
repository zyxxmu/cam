from datasets import load_dataset, load_from_disk
import json
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig ,LlamaTokenizer
import argparse

def get_data():
    dataset=load_dataset(args.dataset,'1.0.0',split='test')
    requests = []
    input_path='./data/xsum_0shot.jsonl'
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))
                break
    request=requests[0]
    num_sample=1000
    file =open('./data/test.jsonl','w')
    for i in range(num_sample):
        temp=request
        temp['article']=str('###\nArticle: '+dataset[i]['article'])
        temp['summary_gt']=dataset[i]['highlights']
        file.write(json.dumps(temp)+'\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    args = parser.parse_args()
    get_data()
