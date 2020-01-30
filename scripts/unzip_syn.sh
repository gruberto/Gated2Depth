#!/bin/bash

download_root=$1
dst="data/syn"

mkdir -p $dst

for item in $download_root/*.zip
do
	unzip $item -d $dst
done
