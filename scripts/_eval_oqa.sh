#!/bin/bash

judge="gpt-4o"

ckpt=$2

CUDA_VISIBLE_DEVICES=$1 python vittle/eval/model_vqa.py \
    --model-path ./checkpoints/${ckpt} \
    --question-file ./playground/data/eval/llava-bench-coco/qa90_questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-coco/val2014 \
    --answers-file ./playground/data/eval/llava-bench-coco/answers/${ckpt}-clean.jsonl \
    --temperature 0

# CUDA_VISIBLE_DEVICES=$1 python vittle/eval/model_vqa.py \
#     --model-path ./checkpoints/${ckpt} \
#     --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${ckpt}-clean.jsonl \
#     --temperature 0

# CUDA_VISIBLE_DEVICES=$1 python vittle/eval/model_vqa_wilder_huggingface.py \
#     --model-path ./checkpoints/${ckpt} \
#     --answers-file ./playground/data/eval/wilder/answers/${ckpt}-clean.jsonl \
#     --temperature 0

# CUDA_VISIBLE_DEVICES=$1 python vittle/eval/model_vqa_wvbench_huggingface.py \
#     --model-path ./checkpoints/${ckpt} \
#     --answers-file ./playground/data/eval/wildvision-bench/answers/${ckpt}-clean.jsonl \
#     --temperature 0


for run in 1 2 3
do

runflag="$run{run}"

python vittle/eval/eval_gpt_review_visual.py \
    --image-path ./playground/data/eval/llava-bench-coco/val2014 \
    --question ./playground/data/eval/llava-bench-coco/qa90_questions.jsonl \
    --context vittle/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    ./playground/data/eval/llava-bench-coco/answers/qa90_gpt4_answer.jsonl \
    ./playground/data/eval/llava-bench-coco/answers/${ckpt}-clean.jsonl \
    --rule vittle/eval/table/rule.json \
    --judge ${judge} \
    --output ./playground/data/eval/llava-bench-coco/reviews/${ckpt}-${judge}-${runflag}.jsonl

python vittle/eval/summarize_gpt_review.py \
    -f ./playground/data/eval/llava-bench-coco/reviews/${ckpt}-${judge}-${runflag}.jsonl \
    --dataset_name llava-bench-coco \
    --judge ${judge} \
    --wb-pj-name OQA \
    --wb-run-name ${ckpt}
done