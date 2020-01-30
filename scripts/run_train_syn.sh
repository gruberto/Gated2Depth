#!/bin/bash

daytime="night" # or "day"
data_dir="./data/syn"
model_dir="./models/model_syn_${daytime}"
results_dir="./results/syn_${daytime}"
train_files="./splits/syn_train_${daytime}.txt"
eval_files="./splits/syn_val_${daytime}.txt"
gpu=0

python src/train_eval.py \
    --results_dir $results_dir \
    --model_dir $model_dir \
    --train_files_path $train_files \
    --eval_files_path $eval_files \
    --base_dir $data_dir \
    --data_type synthetic \
    --gpu $gpu \
    --mode train \
    --num_epochs 2 \
    --lrate 0.0001 \
    --smooth_weight 0.5
