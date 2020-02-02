#!/bin/bash

download_root=$1
dst="data/real"

files=(
	$download_root/depth_hdl64_0.zip
	$download_root/depth_hdl64_1.zip
	$download_root/depth_hdl64_2.zip
	$download_root/depth_hdl64_3.zip
	$download_root/depth_hdl64_4.zip
	$download_root/depth_hdl64_gated_compressed_0.zip
	$download_root/depth_hdl64_gated_compressed_1.zip
	$download_root/depth_hdl64_gated_compressed_2.zip
	$download_root/depth_hdl64_gated_compressed_3.zip
	$download_root/depth_hdl64_gated_compressed_4.zip
	$download_root/depth_hdl64_rgb_left_compressed_0.zip
	$download_root/depth_hdl64_rgb_left_compressed_1.zip
	$download_root/depth_hdl64_rgb_left_compressed_2.zip
	$download_root/depth_hdl64_rgb_left_compressed_3.zip
	$download_root/depth_hdl64_rgb_left_compressed_4.zip
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
	$download_root/rgb_left_8bit_0.zip
	$download_root/rgb_left_8bit_1.zip
	$download_root/rgb_left_8bit_2.zip
	$download_root/rgb_left_8bit_3.zip
	$download_root/rgb_left_8bit_4.zip
	$download_root/rgb_right_8bit_0.zip
	$download_root/rgb_right_8bit_1.zip
	$download_root/rgb_right_8bit_2.zip
	$download_root/rgb_right_8bit_3.zip
	$download_root/rgb_right_8bit_4.zip
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
