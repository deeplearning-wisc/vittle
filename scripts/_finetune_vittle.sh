#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"

ibl_idx=layer-24
alpha=0.5
beta=0.1

for lp in F L
do

deepspeed --master_port=29501 vittle/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/LLaVA-Instruct/llava_v1_5_mix665k.json \
    --image_folder ./playground/data/LLaVA-Instruct \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "./checkpoints/vittle-7b-${lp}-${ibl_idx}-a${alpha}-b${beta}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --bottleneck_layeridx_t $ibl_idx \
    --bottleneck_layeridx_v $ibl_idx \
    --ib_strength_v $beta \
    --ib_strength_t $beta \
    --ib_fadein_end $alpha \
    --learnable_prior_flag $lp \
    --report_to wandb \
    --run_name "vittle-7b-${lp}-${ibl_idx}-a${alpha}-b${beta}"

done