import json
import os
from collections import defaultdict

import numpy as np

import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-d', '--dir', default=None)
    parser.add_argument('-v', '--version', default=None)
    parser.add_argument('-s', '--select', nargs='*', default=None)
    parser.add_argument('-f', '--files', nargs='*', default=[])
    parser.add_argument('-i', '--ignore', nargs='*', default=[])
    
    parser.add_argument("--wb-pj-name", type=str, default='')
    parser.add_argument("--wb-run-name", type=str, default='')
    parser.add_argument("--judge", type=str, default='gpt-4')
    parser.add_argument("--dataset_name", type=str, default='llava-bench-coco')
    parser.add_argument("--output_path", type=str, default='vittle/results_db')
    parser.add_argument("--output_filename", type=str, default='test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.wb_pj_name:
        import wandb
        wandb.init(
        project=args.wb_pj_name,
        name=args.wb_run_name if args.wb_run_name else None,
        config=vars(args),)

    if args.ignore is not None:
        args.ignore = [int(x) for x in args.ignore]

    if len(args.files) > 0:
        review_files = args.files
    else:
        review_files = [x for x in os.listdir(args.dir) if x.endswith('.jsonl') and (x.startswith('gpt4_text') or x.startswith('reviews_') or x.startswith('review_') or 'review' in args.dir)]

    for review_file in sorted(review_files):
        config = os.path.basename(review_file).replace('gpt4_text_', '').replace('.jsonl', '')
        if args.select is not None and any(x not in config for x in args.select):
            continue
        if '0613' in config:
            version = '0613'
        else:
            version = '0314'
        if args.version is not None and args.version != version:
            continue
        scores = defaultdict(list)
        print(config)
        with open(os.path.join(args.dir, review_file) if args.dir is not None else review_file) as f:
            for review_str in f:
                review = json.loads(review_str)
                
                if review['question_id'] in args.ignore:
                    continue

                if args.idx_path:
                    import pickle
                    with open(args.idx_path, "rb") as file:
                        idx_dict = pickle.load(file)
                    if review['question_id'] not in idx_dict[args.indexing_basis]:
                        continue

                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])
        perf_dict = {}
        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            # print(k, stats, round(stats[1]/stats[0]*100, 1))
            print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
            perf_dict[f'{k}_ratio'] = round(stats[1]/stats[0]*100, 1)
            perf_dict[f'{k}_model0'] = round(stats[0] * 10, 1)
            perf_dict[f'{k}_model1'] = round(stats[1] * 10, 1)

        try:
            perf_dict.update(args)
            os.makedirs(args.output_path, exist_ok=True)
            csv_path = os.path.join(args.output_path, args.output_filename+".csv")
            # Write or append to CSV
            write_header = not os.path.exists(csv_path)
            with open(csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=perf_dict.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(perf_dict)
        except Exception as error:
            print("An exception occurred:", error)

        
        if args.wb_pj_name:
            wandb.log(perf_dict)
        print('=================================')
