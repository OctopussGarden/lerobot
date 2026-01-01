#!/bin/bash

# Training command
OUTPUT_DIR="./outputs/pi05_training_$(date +%Y-%m-%d_%H-%M-%S)"
python ./src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=xuweiwu/bimanual-toy-box-cleanup \
    --policy.type=pi05 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=pi05_training \
    --policy.repo_id=octo/pi05_finetuned_toy_box_cleanup \
    --policy.pretrained_path=models/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.use_amp=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=100 \
    --policy.device=cuda \
    --batch_size=1 \
