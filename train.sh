#!/bin/bash 
module load cuda/10.1
module load anaconda
source activate pool 

:<<!EOF!
https://cloud.seatable.cn/workspace/126932/dtable/%E5%A4%9AGPU%E5%AE%9E%E9%AA%8C%E5%AE%89%E6%8E%92/?tid=1FE3&vid=0000
TODO LIST:
1. mobilenetv2_100 训练到72+, 目前只有71 [4卡运行] running ....
2. 按照poolformer的配置训练rfformer [4卡运行] running....
3. resnet系列原版训练好, 先去搞定resnet50 [8卡运行] running....
!EOF!

######################## poolformer rfformer #######################

# MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
# DROP_PATH=0. 
# drop path rates [0.1, 0.1, 0.2, 0.3, 0.4] responding to model [s12, s24, s36, m36, m48]
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /data/public/imagenet-mini --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp

# RUNNING....FAILED [decrease learning rate] RUNNING...DONE
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 33844 /data/public/imagenet2012 --model rfformer_s12 -b 256 --lr 4e-4 --drop-path 0.1 --apex-amp --sched 'adamw' --weight-decay 0.05 --epochs 90 

# RUNNING...
# 作为对比运行poolformer
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 33844 /data/public/imagenet2012 --model poolformer_s12 -b 256 --lr 4e-4 --drop-path 0.1 --apex-amp --sched 'adamw' --weight-decay 0.05 --epochs 90 

############################ mobilenetv2系列 #################################

# 参考https://github.com/tonylins/pytorch-mobilenet-v2 提供的配置
# RUNNING....FAILED....RUNNING....DONE
# 期望结果: 72+ 实际结果：71.05
# 15小时，150 epochs 
# output/train/20220224-225223-mobilenetv2_100-224
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 33492 /data/public/imagenet2012 \
#   --model mobilenetv2_100 -b 128 --lr 0.05 --drop-path 0.  --apex-amp --weight-decay 4e-5 --epochs 150 --sched 'cosine'


# 再次运行，尝试用model_ema, longer training 200 epoch 
# RUNNING...
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 32492 /data/public/imagenet2012 \
#   --model mobilenetv2_100 -b 128 --lr 0.01 --drop-path 0.  --apex-amp --weight-decay 4e-5 --epochs 200 --sched 'cosine' --model-ema --model-ema-decay 0.9999


################################ resnet 系列 ################################### 

# RUNNING....DONE
# output/train/20220224-100917-resnet18-224
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 19231 /data/public/imagenet2012 \
#   --model resnet18 -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 50 --decay-epochs 15 | tee "./output/resnet18.log"

# RUNNING....DONE
# output/train/20220224-101010-resnet18_rf-224
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 37283 /data/public/imagenet2012 \
#   --model resnet18_rf -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 50 --decay-epochs 15 | tee "./output/resnet18_rf.log"

# RUNNING....DONE
# 期望结果：比resnet18高 [已证实，不过当时设计的是每一层都添加rfsa，不太合适]
# output/train/20220224-153529-resnet18_rfsa-224
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 39271 /data/public/imagenet2012 \
#   --model resnet18_rfsa -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 50 --decay-epochs 15 | tee "./output/resnet18_rfsa.log"

# RUNNING....FAILED....RUNNING....FAILED
# 修改lr from 0.1 to 1e-3
# 期望结果：71+ FAILED
# 参考：https://github1s.com/PaddlePaddle/PaddleClas 的配置
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 29930 /data/public/imagenet2012 \
#   --model resnet18 -b 64 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 120 --decay-epochs 15 --smoothing 0 --opt 'momentum' --sched "cosine" --weight-decay 0.05 

# RUNNING....FAILED....RUNNING....FAILED
# 修改lr from 0.1 to 1e-3
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 29901 /data/public/imagenet2012 \
#   --model resnet18_rf -b 64 --lr 1e-3 --drop-path $DROP_PATH --apex-amp --epochs 120 --decay-epochs 15 --smoothing 0 --opt 'momentum' --sched "cosine" --weight-decay 0.0001

# 参考蒋神配置
# running resnet50
# RUNNING...
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 29201 /data/public/imagenet2012 \
#   --model resnet50 -b 512 --lr 0.2 --drop-path 0. --apex-amp --epochs 90 --decay-epochs 30 --opt sgd --sched step --weight-decay 0.0001 

# RUNNING....
# resnet50_rfsa
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 32241 /data/public/imagenet2012 \
  # --model resnet50_rfsa -b 256 --lr 0.1 --drop-path 0. --apex-amp --epochs 90 --decay-epochs 30 --opt sgd --sched step --weight-decay 0.0001 


# ############# 
# DATA=/data/home/scv6681/run/data/imagenet20_percent
# CUDA_VISIBLE_DEVICES=0 ./distributed_train.sh 1 29911 $DATA \
#   --model resnet18 -b 64 --lr 1e-3 --drop-path 0. --apex-amp --epochs 120 --decay-epochs 15 --smoothing 0 --opt 'momentum' --sched "cosine" --weight-decay 0.0001 


################################################# 20 % imagenet ####################
DATA=/data/home/scv6681/run/data
CUDA_VISIBLE_DEVICES=0 ./distributed_train.sh 1 29301 $DATA \
  --model resnet50 -b 256 --lr 4e-3 --drop-path 0. --apex-amp --epochs 100 --decay-epochs 30 --opt lamb --sched cosine --weight-decay 0.02 --reprob 0. --color-jitter 0. 

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 29211 $DATA \
#   --model resnet50_rf -b 256 --lr 4e-3 --drop-path 0. --apex-amp --epochs 100 --decay-epochs 30 --opt lamb --sched cosine --weight-decay 0.02 --reprob 0. --color-jitter 0. 