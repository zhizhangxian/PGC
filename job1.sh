#!/bin/bash
#BSUB -q gpu_v100
#BSUB -m gpu22
#BSUB -gpu num=1
CUDA_VISIBLE_DEVICES=6 python New_main.py --model deeplabv3plus_resnet101 --gpu_id 6 --year 2012_aug --crop_val  --crop_size 513 --batch_size 16 --output_stride 16

