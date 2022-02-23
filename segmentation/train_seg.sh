#!/bin/bash 
module load cuda/10.1
module load anaconda
source activate pool 


bash dist_train.sh configs/sem_fpn/PoolFormer/fpn_poolformer_s12_ade20k_40k.py 1

