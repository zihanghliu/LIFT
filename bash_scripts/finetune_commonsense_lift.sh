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
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_lift_commonsense.txt)
MODEL=$(echo $cfg | cut -f 1 -d ' ')
no_grad=$(echo $cfg | cut -f 2 -d ' ')
mask=$(echo $cfg | cut -f 3 -d ' ')
lr=$(echo $cfg | cut -f 4 -d ' ')
lora_rank=$(echo $cfg | cut -f 5 -d ' ')
filter_rank=$(echo $cfg | cut -f 6 -d ' ')
update_interval=$(echo $cfg | cut -f 7 -d ' ')
seed=$(echo $cfg | cut -f 8 -d ' ')

echo $MODEL


peft_tuner=sparse


OUTPUT=${OUTPUT_SRC_DIR}/${MODEL}/lift/commonsense/${peft_tuner}_no_${no_grad}_mask_${mask}_rank_${lora_rank}_filter_${filter_rank}_interval_${update_interval}/lr_${lr}/seed_${seed}
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./outs/math/s2_llama3
fi
mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
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
    --peft_tuner ${peft_tuner} \
    --mask_type ${mask} \
    --lora_rank ${lora_rank} \
    --filter_rank ${filter_rank} \
    --update_interval ${update_interval} \
    --save_interval 5000 \
    --instruction_type single \
    --val_set_size 120 \
    --eval_step 400 \
    --no_grad ${no_grad} \
    --data_path ${DATA_DIR}/LLM-Adapters/ft-training_set/commonsense_170k.json \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

bash ./bash_scripts/eval_commonsense.sh $OUTPUT