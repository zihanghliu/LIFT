#!/bin/bash

MODEL=$1
OUTPUT_DIR=$MODEL/math

SRC_DIR=/enter/your/path/to/the/repo
DATA_DIR=/enter/your/data/dir

datasets=(MultiArith gsm8k AddSub AQuA SingleEq SVAMP mawps)

cd $SRC_DIR

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=32
    accelerate launch --main_process_port $master_port ./src/eval/run_math_parallel.py \
        --data_path ${DATA_DIR}/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dtype bf16 \
        --dataset $dataset \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
done


