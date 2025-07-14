#!/bin/bash

adapter_name=$1
base_model=$2
MODEL=$3
OUTPUT_DIR=$MODEL/commonsense

SRC_DIR=/enter/your/path/to/the/repo
DATA_DIR=/enter/your/data/dir

if [ "$OUTPUT_DIR" == "" ]; then
    OUTPUT_DIR=./results/$MODEL/commonsense
fi

datasets=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)

cd $SRC_DIR

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=16
    accelerate launch --main_process_port $master_port eval/run_commonsense_parallel.py \
        --data_path ${DATA_DIR}/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dataset $dataset \
        --adapter_name $adapter_name \
        --base_model $base_model \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
done