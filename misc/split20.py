import os 
import random 
import shutil 

root = "/data/public/imagenet2012/train"
dstn = "/data/home/scv6681/run/data/imagenet20_percent"


for cls_file in os.listdir(root):
    print(f"processing {cls_file}")
    cls_full_path = os.path.join(root, cls_file)
    n = len(os.listdir(cls_full_path))
    rate = 0.2 
    picknumber = int(n * rate)
    sampled = random.sample(os.listdir(cls_full_path), picknumber)
    
    for single in sampled:
        dstn_cls = os.path.join(dstn, cls_file)
        if not os.path.exists(dstn_cls):
            os.makedirs(dstn_cls)
        shutil.copy(os.path.join(cls_full_path, single), os.path.join(dstn_cls, single))