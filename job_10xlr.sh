#!/bin/bash
#BSUB -J 10xlr
#BSUB -q gpu_v100
#BSUB -m gpu18
#BSUB -gpu num=1
CUDA_VISIBLE_DEVICES=0 python main.py --model deeplabv3plus_resnet101 --lr 0.25 --gpu_id 0 --year 2012_aug --crop_val  --batch_size 16 --output_stride 16

