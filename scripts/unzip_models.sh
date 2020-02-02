#!/bin/bash

download_root=$1
dst="models"

files=(
	$download_root/gated2depth_real_day.zip
	$download_root/gated2depth_real_night.zip
	$download_root/gated2depth_syn_day.zip
	$download_root/gated2depth_syn_night.zip
)
 
mkdir -p $dst

all_exists=true
for item in ${files[*]} 
do
	if [[ ! -f "$item" ]]; then
    		echo "$item is missing"
		all_exists=false
	fi
done

if $all_exists; then
	for item in ${files[*]} 
	do
		unzip $item -d $dst
	done
fi
