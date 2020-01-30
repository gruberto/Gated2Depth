#!/bin/bash

daytime="night" # or "day"
data_dir="./data/real"
model_dir="./models/model_real_${daytime}"
results_dir="./results/real_${daytime}"
train_files="./splits/real_train_${daytime}.txt"
eval_files="./splits/real_val_${daytime}.txt"
gpu=0

python src/train_eval.py \
    --results_dir $results_dir \
    --model_dir $model_dir \
    --train_files_path $train_files \
    --eval_files_path $eval_files \
    --base_dir $data_dir \
    --data_type real \
    --gpu $gpu \
    --mode train \
    --num_epochs 2 \
    --lrate 0.0001 \
    --smooth_weight 0.5
