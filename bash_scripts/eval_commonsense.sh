#!/bin/bash

MODEL=$1
OUTPUT_DIR=$MODEL/commonsense

SRC_DIR=/enter/your/path/to/the/repo
DATA_DIR=/enter/your/data/dir

datasets=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)

cd $SRC_DIR

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=32
    accelerate launch --main_process_port $master_port eval/run_commonsense_parallel.py \
        --data_path ${DATA_DIR}/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dtype bf16 \
        --dataset $dataset \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
done


