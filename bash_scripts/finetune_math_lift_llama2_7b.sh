#!/bin/bash

pwd
hostname
date
echo starting job...
conda activate lift
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

SRC_DIR=/enter/your/path/to/the/repo
DATA_DIR=/enter/your/data/dir
OUTPUT_SRC_DIR=/enter/your/output/dir

MODEL=meta-llama/Llama-2-7b-hf
no_grad=head
mask=weight_filtered_mag_abs_largest_sparse
lr=1e-4
lora_rank=256
filter_rank=64
update_interval=400
seed=43

echo $MODEL

peft_tuner=sparse

OUTPUT=${OUTPUT_SRC_DIR}/${MODEL}/lift/math/${peft_tuner}_no_${no_grad}_mask_${mask}_rank_${lora_rank}_filter_${filter_rank}_interval_${update_interval}/lr_${lr}/seed_${seed}

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --peft_tuner ${peft_tuner} \
    --mask_type ${mask} \
    --lora_rank ${lora_rank} \
    --filter_rank ${filter_rank} \
    --update_interval ${update_interval} \
    --load_last_model \
    --no_grad ${no_grad} \
    --data_path ${DATA_DIR}/LLM-Adapters/ft-training_set/math_10k.json \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

bash ./bash_scripts/eval_math.sh $OUTPUT