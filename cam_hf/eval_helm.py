from rouge import Rouge
import json
import argparse


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []
    cover_list=[]
    input_path=args.input_path
    prompt_list=[]
    labels=[]
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                result=json.loads(line)
                prompt=result['request']['article']
                prompt_list.append(prompt)
                label=result["request"]["summary_gt"]
                generate_text=result["result"]["choices"][0]["text"]
                
                ground=set(label)
                gen=set(generate_text)
                cover=ground.intersection(gen)
                coverage= len(cover)/len(ground)
                cover_list.append(coverage)
                hyp=generate_text
                hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
                if len(hyp)<=0:
                    rouge1_score_list.append(0)
                    rouge2_score_list.append(0)
                    rougel_score_list.append(0)
                    continue
                scores = rouge.get_scores(generate_text, label)[0]
                labels.append(label)
                rouge1_score_list.append(scores['rouge-1']['f'])
                rouge2_score_list.append(scores['rouge-2']['f'])
                rougel_score_list.append(scores['rouge-l']['f'])
    r1=sum(rouge1_score_list)/len(rouge1_score_list)
    r2=sum(rouge2_score_list)/len(rouge1_score_list)
    rl=sum(rougel_score_list)/len(rouge1_score_list)
    cover=sum(cover_list)/len(cover_list)
    print("rouge 1 :",r1)
    print("rouge 2 :",r2)
    print("rouge l :",rl)
    print("coverage :",cover)
    with open(args.output_path,'w') as file:
        file.write("the task is :"+args.task+'\n')
        file.write("the method is :"+args.method+'\n')
        file.write("the model is :"+args.model_name+'\n')
        file.write("the rouge1 is :"+str(r1)+'\n')
        file.write("the rouge2 is :"+str(r2)+'\n')
        file.write("the rougel is :"+str(rl)+'\n')
        file.write("the coverage is :"+str(cover)+'\n')
        file.write("****************************************************\n")
if __name__ == "__main__":
    test()
    