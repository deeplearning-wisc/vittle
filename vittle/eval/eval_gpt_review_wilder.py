import argparse
import json
import os

import openai
from openai import AzureOpenAI
from openai import OpenAI
import time
import pdb
from tqdm import tqdm
import base64

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

NUM_SECONDS_TO_SLEEP = 0.01

def get_eval(content: str, max_tokens: int, judge='gpt-4o', api='v1'):
    while True:
        try:
            deployment_name = 'gpt-4o'
            api_ver = '2025-01-01-preview'
            
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_4O_2025")
            apikey = os.getenv("AZURE_OPENAI_API_KEY_2025")
            

            client = AzureOpenAI(
                    azure_endpoint=endpoint, 
                    api_key=apikey,  
                    api_version=api_ver
                    )
            messages=[{
                'role': 'system',
                'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
            }, {
                'role': 'user',
                'content': content,
            }]
            response = client.chat.completions.create(model=deployment_name, messages=messages, max_tokens=max_tokens)
            break
        except Exception as e:
            print(e)
    
    time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--judge', type=str, default='gpt-4o')
    parser.add_argument('--api', type=str, default='v1')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/LLaVA-Bench-Wilder", split="test")

    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []
    
    review_file = open(f'{args.output}', 'a')


    handles = []
    idx = 0
    rule = rule_dict['wildvision_bench']
    prompt = rule['prompt']
    role = rule['role']

    for ans1_js in tqdm(f_ans1):
        ans1 = json.loads(ans1_js)
        qid = ans1['question_id']

        encoded_img = ds[idx]["image"]
        content = (f'[Image]\n{encoded_img}\n\n'
                   f'[Question]\n{ans1["prompt"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans1["gt_answer"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        
        cur_js = {
            'id': idx+1,
            'question_id': ans1['question_id'],
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens, args.judge, args.api)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        
        idx += 1
        print(idx)
    review_file.close()
