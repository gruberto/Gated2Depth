#!/bin/bash

download_root=$1
dst="data/syn"

files=(
	$download_root/depth_compressed_0.zip
	$download_root/depth_compressed_1.zip
	$download_root/depth_compressed_2.zip
	$download_root/depth_compressed_3.zip
	$download_root/depth_compressed_4.zip
	$download_root/gated0_10bit_0.zip
	$download_root/gated0_10bit_1.zip
	$download_root/gated0_10bit_2.zip
	$download_root/gated0_10bit_3.zip
	$download_root/gated0_10bit_4.zip
	$download_root/gated1_10bit_0.zip
	$download_root/gated1_10bit_1.zip
	$download_root/gated1_10bit_2.zip
	$download_root/gated1_10bit_3.zip
	$download_root/gated1_10bit_4.zip
	$download_root/gated2_10bit_0.zip
	$download_root/gated2_10bit_1.zip
	$download_root/gated2_10bit_2.zip
	$download_root/gated2_10bit_3.zip
	$download_root/gated2_10bit_4.zip
	$download_root/rgb_left_0.zip
	$download_root/rgb_left_1.zip
	$download_root/rgb_left_2.zip
	$download_root/rgb_left_3.zip
	$download_root/rgb_left_4.zip
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
