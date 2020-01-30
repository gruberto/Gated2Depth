#!/bin/bash

split="test" # or "val"
data_dir="./data/real"
model_dir="./models/gated2depth_real_night/model.ckpt-8028"
results_dir="./results/gated2depth_real_night/${split}"
eval_files="./splits/real_${split}_night.txt"
gpu=0

python src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval

