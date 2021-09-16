#!/bin/bash
#BSUB -J mode_3
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process"
python main.py --alpha 0.3 --lr 0.1 --total_itrs 30000 --model deeplabv3plus_resnet101 --gpu_id 3 --year 2012_aug --crop_val  --batch_size 16 --output_stride 16 --pgc_mode 0

