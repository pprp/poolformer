#!/bin/bash 
module load cuda/10.1
module load anaconda
source activate pool 


# MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
MODEL=mobilenetv2_100
DROP_PATH=0. # drop path rates [0.1, 0.1, 0.2, 0.3, 0.4] responding to model [s12, s24, s36, m36, m48]
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /data/public/imagenet-mini \
#   --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp
CUDA_VISIBLE_DEVICES=0,1 ./distributed_train.sh 2 /data/public/imagenet-mini \
  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp | tee "./output/mobilenetv2.log"