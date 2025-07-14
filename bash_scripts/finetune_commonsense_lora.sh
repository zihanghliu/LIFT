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
export HF_HOME=/your/path/to/huggingface/cache      # MODIFY THIS LINE

SRC_DIR=/enter/your/path/to/the/repo      # MODIFY THIS LINE
DATA_DIR=/enter/your/data/dir      # MODIFY THIS LINE
OUTPUT_SRC_DIR=/enter/your/output/dir      # MODIFY THIS LINE

SLURM_ARRAY_TASK_ID=$1
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_lora_commonsense.txt)
MODEL=$(echo $cfg | cut -f 1 -d ' ')
adapter_name=$(echo $cfg | cut -f 2 -d ' ')
lr=$(echo $cfg | cut -f 3 -d ' ')
lora_r=$(echo $cfg | cut -f 4 -d ' ')
lora_alpha=$(echo $cfg | cut -f 5 -d ' ')
seed=$(echo $cfg | cut -f 6 -d ' ')


OUTPUT=${OUTPUT_SRC_DIR}/${MODEL}/${adapter_name}/commonsense/lr_${lr}/rank_${lora_r}_alpha_${lora_alpha}/seed_${seed}
mkdir -p $OUTPUT

cd $SRC_DIR

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_lora.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --load_last_model \
    --adapter_name ${adapter_name} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --data_path ${DATA_DIR}/LLM-Adapters/ft-training_set/commonsense_170k.json \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

bash ./bash_scripts/eval_commonsense_lora.sh ${adapter_name} $MODEL $OUTPUT