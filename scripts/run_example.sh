#!/bin/bash

data_dir="./example"
results_dir="./results/example"
gpu=0

model_dir="./models/gated2depth_real_night/model.ckpt-8028"

python src/train_eval.py \
    --results_dir $results_dir/night \
	  --model_dir $model_dir \
	  --eval_files_path ./splits/example_night.txt \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --show_result

model_dir="./models/gated2depth_real_day/model.ckpt-13460"

python src/train_eval.py \
    --results_dir $results_dir/day \
	  --model_dir $model_dir \
	  --eval_files_path ./splits/example_day.txt \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --show_result

