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


exp_no="scalebio_flywheel_nowarmup"

lr=1e-6
epoch=2
mode=bilevel

exp_id=finetune_${exp_no}_epoch-${epoch}_lr-$lr

project_dir=/u/xpan2/projects/scalebio/LLM-Finetuning/
# project_dir=$(cd "$(dirname $0)"; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}
output_dir=${project_dir}/models/${exp_id}
data_path=${project_dir}/data/${exp_no}/${mode}

echo ${project_dir}

accelerate launch --config_file fsdp_config.yaml python/train.py \
    --model $model_and_tokenizer \
    --tokenizer-name $model_and_tokenizer \
    --train-data ${data_path}/train/\*.json \
    --val-data ${data_path}/val/\*.json \
    --optimizer "name=adamw, lr=${lr}, weight_decay=5e-4" \
    --bf16 \
    --warmup-ratio 0.03 \
    --pseudo_random \
    --logging_conf_file ${project_dir}/conf/common.log_conf \
    --seed 1234 \
    --max-length 1024 \
    --epoch ${epoch} \
    --val_batch_size 2 \
    --eval_frequency 10 \
    --response_loss_only \
    --save_dir ${output_dir} \
    --global_batch_size 64 \
    --lmflow-format \
    --micro_batch_size 4 \
    | tee ${log_dir}/train.log \
        2> ${log_dir}/train.err