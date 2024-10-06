#!/bin/bash
#SBATCH --job-name="finetune llama3.1 flywheel"
#SBATCH  --account=bckr-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=512g
#SBATCH --time=10:59:00
#SBATCH --output="run_finetune_llama3.1.log"
#SBATCH --error="run_finetune_llama3.1.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

# model_and_tokenizer=Qwen/Qwen2-7B
# model_and_tokenizer=google/gemma-2-9b
model_and_tokenizer=meta-llama/Meta-Llama-3.1-8B
# model_and_tokenizer=openai-community/gpt2

accelerate launch --config_file fsdp_config.yaml python/train.py \
    --model $model_and_tokenizer \
    --tokenizer-name $model_and_tokenizer \
    --train-data /u/xpan2/projects/scalebio/dev_accelerate/flywheel_sample_5k/sharegpt_format/train_\*.json \
    --val-data /u/xpan2/projects/scalebio/dev_accelerate/flywheel_sample_5k/sharegpt_format/val.json \
    --optimizer "name=adamw, lr=1e-5, weight_decay=5e-4" \
    --bf16 \
    --warmup-ratio 0.03 \
    --pseudo_random \
    --logging_conf_file conf/common.log_conf \
    --seed 1234 \
    --max-steps 200 \
    --max-length 1024 \
    --epoch 1 \
    --val_batch_size 2 \
    --eval_frequency 50 \
    --response_loss_only \
    --save_dir ./llama3.1-flywheel-warmup/ \
    --global_batch_size 64 \
    --sharegpt_format \
    --micro_batch_size 4 \
    > flywheel-llama3.1.log \
    2> flywheel-llama3.1.err