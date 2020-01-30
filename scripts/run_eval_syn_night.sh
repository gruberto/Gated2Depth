#!/bin/bash

split="test" # or "val"
data_dir="./data/syn"
model_dir="./models/gated2depth_syn_night/model.ckpt-38850"
results_dir="./results/gated2depth_syn_night/${split}"
eval_files="./splits/syn_${split}_night.txt"
gpu=0

python src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type synthetic \
	  --gpu $gpu \
	  --mode eval

