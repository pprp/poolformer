
import os 
import sys 

from thop import profile
from ptflops import get_model_complexity_info
import torch 


sys.path.append("..")

from models.autorf.components import *
from models.autorf.operations import * 


i = torch.zeros(3, 32, 64, 64)


m1 = ReceptiveFieldAttention(32, genotype=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_5x5', 1), ('max_pool_3x3', 0), ('noise', 2), ('noise', 1)], normal_concat=range(0, 4)))
m2 = ReceptiveFieldSelfAttention(32, genotype=Genotype(normal=[('strippool', 0), ('avg_pool_3x3', 0), ('avg_pool_5x5', 1), ('avg_pool_7x7', 0), ('strippool', 2), ('noise', 1)], normal_concat=range(0, 4)))
m3 = SE(32, reduction=4)
m4 = SPP(32, 32)
m5 = CMlp(32)
m6 = StripPool(32)
m7 = CBAM(32)



print("rf:",sum([param.nelement() for param in m1.parameters()])) # 2752
print("rfsa:",sum([param.nelement() for param in m2.parameters()])) # 3664
print("se:",sum([param.nelement() for param in m3.parameters()])) # 512 
print("spp:",sum([param.nelement() for param in m4.parameters()])) # 2656
print("cmlp:",sum([param.nelement() for param in m5.parameters()])) # 2112 
print("strippool:",sum([param.nelement() for param in m6.parameters()])) # 1776 
print("cbam:",sum([param.nelement() for param in m7.parameters()])) # 262 

# flops, params = profile(m1, torch.randn(2, 32, 64, 64))


# get_model_complexity_info(m1, (32, 224,224), as_strings=True, print_per_layer_stat=True)