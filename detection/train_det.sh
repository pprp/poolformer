#!/bin/bash 
module load cuda/10.1
module load anaconda
source activate pool 


FORK_LAST3=1 bash dist_train.sh configs/retinanet_poolformer_s12_fpn_1x_coco.py 1

