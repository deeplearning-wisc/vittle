#!/bin/bash

ckpt=$2

CUDA_VISIBLE_DEVICES=$1 python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/${ckpt} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${ckpt}-clean.jsonl \
    --temperature 0 \
    --conv-mode llava_v1

python vittle/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${ckpt}-clean.jsonl \
    --wb-pj-name HAL \
    --wb-run-name ${ckpt}