#!/bin/bash 
module load cuda/10.1
module load anaconda
source activate pool 



# MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
# DROP_PATH=0. 
# drop path rates [0.1, 0.1, 0.2, 0.3, 0.4] responding to model [s12, s24, s36, m36, m48]
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /data/public/imagenet-mini --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp

CUDA_VISIBLE_DEVICES=0 ./distributed_train.sh 1 33844 /data/public/imagenet-mini --model rfformer_s12 -b 128 --lr 1e-3 --drop-path 0.1 --apex-amp --sched 'adamw' --weight-decay 0.05

############################ mobilenetv2系列 #################################

# 参考https://github.com/tonylins/pytorch-mobilenet-v2 提供的配置
# RUNNING....FAILED....RUNNING....
# 期望结果: 72+
# 8小时，84 epoch RUNNING
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 33492 /data/public/imagenet2012 \
#   --model mobilenetv2_100 -b 128 --lr 0.05 --drop-path 0.  --apex-amp --weight-decay 4e-5 --epochs 150 --sched 'cosine'

# CUDA_VISIBLE_DEVICES=0 ./distributed_train.sh 1 /data/public/imagenet-mini \
#   --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp | tee "./output/test.log"


################################ resnet 系列 ################################### 

# RUNNING....DONE
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 19231 /data/public/imagenet2012 \
#   --model resnet18 -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 50 --decay-epochs 15 | tee "./output/resnet18.log"

# RUNNING....DONE
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 37283 /data/public/imagenet2012 \
#   --model resnet18_rf -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 50 --decay-epochs 15 | tee "./output/resnet18_rf.log"

# RUNNING....DONE
# 期望结果：比resnet18高 [已证实，不过当时设计的是每一层都添加rfsa，不太合适]
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 39271 /data/public/imagenet2012 \
#   --model resnet18_rfsa -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 50 --decay-epochs 15 | tee "./output/resnet18_rfsa.log"

# RUNNING....FAILED....RUNNING....FAILED
# 期望结果：71+
# 参考：https://github1s.com/PaddlePaddle/PaddleClas 的配置
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 29930 /data/public/imagenet2012 \
#   --model resnet18 -b 64 --lr 0.1 --drop-path $DROP_PATH --apex-amp --epochs 120 --decay-epochs 15 --smoothing 0 --opt 'momentum' --sched "cosine" --weight-decay 0.05 

# RUNNING....FAILED....RUNNING....FAILED
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 29901 /data/public/imagenet2012 \
#   --model resnet18_rf -b 64 --lr 0.1 --drop-path $DROP_PATH --apex-amp --epochs 120 --decay-epochs 15 --smoothing 0 --opt 'momentum' --sched "cosine" --weight-decay 0.0001

