""" The EfficientNet Family in PyTorch
An implementation of EfficienNet that covers variety of related models with efficient architectures:
* EfficientNet-V2
  - `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent weight ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
  - Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665
  - Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252
* MixNet (Small, Medium, and Large)
  - MixConv: Mixed Depthwise Convolutional Kernels - https://arxiv.org/abs/1907.09595
* MNasNet B1, A1 (SE), Small
  - MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/abs/1807.11626
* FBNet-C
  - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS - https://arxiv.org/abs/1812.03443
* Single-Path NAS Pixel1
  - Single-Path NAS: Designing Hardware-Efficient ConvNets - https://arxiv.org/abs/1904.02877
* And likely more...
The majority of the above models (EfficientNet*, MixNet, MnasNet) and original weights were made available
by Mingxing Tan, Quoc Le, and other members of their Google Brain team. Thanks for consistently releasing
the models and weights open source!
Hacked together by / Copyright 2021 Ross Wightman
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from timm.models.features import FeatureInfo, FeatureHooks
from timm.models.helpers import build_model_with_cfg, default_cfg_for_features
from timm.models.layers import create_conv2d, create_classifier
from timm.models.registry import register_model

__all__ = ['EfficientNet', 'EfficientNetFeatures']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mnasnet_050': _cfg(url=''),
    'mnasnet_075': _cfg(url=''),
    'mnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth'),
    'mnasnet_140': _cfg(url=''),

    'semnasnet_050': _cfg(url=''),
    'semnasnet_075': _cfg(url=''),
    'semnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth'),
    'semnasnet_140': _cfg(url=''),
    'mnasnet_small': _cfg(url=''),

    'mobilenetv2_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth'),
    'mobilenetv2_110d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_110d_ra-77090ade.pth'),
    'mobilenetv2_120d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_120d_ra-5987e2ed.pth'),
    'mobilenetv2_140': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_140_ra-21a4e913.pth'),

    'fbnetc_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
        interpolation='bilinear'),
    'spnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
        interpolation='bilinear'),

    # NOTE experimenting with alternate attention
    'eca_efficientnet_b0': _cfg(
        url=''),
    'gc_efficientnet_b0': _cfg(
        url=''),

    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        test_input_size=(3, 256, 256), crop_pct=1.0),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), test_input_size=(3, 288, 288), crop_pct=1.0),
    'efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), crop_pct=1.0),
    'efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), test_input_size=(3, 384, 384), crop_pct=1.0),
    'efficientnet_b5': _cfg(
        url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'efficientnet_b6': _cfg(
        url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'efficientnet_b7': _cfg(
        url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'efficientnet_b8': _cfg(
        url='', input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
    'efficientnet_l2': _cfg(
        url='', input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.961),

    'efficientnet_es': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth'),
    'efficientnet_em': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_em_ra2-66250f76.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_el': _cfg(
        url='https://github.com/DeGirum/pruned-models/releases/download/efficientnet_v1.0/efficientnet_el.pth', 
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'efficientnet_es_pruned': _cfg(
        url='https://github.com/DeGirum/pruned-models/releases/download/efficientnet_v1.0/efficientnet_es_pruned75.pth'),
    'efficientnet_el_pruned': _cfg(
        url='https://github.com/DeGirum/pruned-models/releases/download/efficientnet_v1.0/efficientnet_el_pruned70.pth', 
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'efficientnet_cc_b0_4e': _cfg(url=''),
    'efficientnet_cc_b0_8e': _cfg(url=''),
    'efficientnet_cc_b1_8e': _cfg(url='', input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),

    'efficientnet_lite0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth'),
    'efficientnet_lite1': _cfg(
        url='',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_lite2': _cfg(
        url='',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'efficientnet_lite3': _cfg(
        url='',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_lite4': _cfg(
        url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),

    'efficientnet_b1_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb1_pruned_9ebb3fe6.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b2_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb2_pruned_203f55bc.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b3_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb3_pruned_5abcc29f.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'efficientnetv2_rw_t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth',
        input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0),
    'gc_efficientnetv2_rw_t': _cfg(
        url='',
        input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0),
    'efficientnetv2_rw_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_v2s_ra2_288-a6477665.pth',
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_rw_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_rw_m_agc-3d90cb1e.pth',
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),

    'efficientnetv2_s': _cfg(
        url='',
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_m': _cfg(
        url='',
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),
    'efficientnetv2_l': _cfg(
        url='',
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'efficientnetv2_xl': _cfg(
        url='',
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth',
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_b0_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ap-f262efe1.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, input_size=(3, 224, 224)),
    'tf_efficientnet_b1_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ap-2f8e7636.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ap-aad25bdd.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ap-dedb23e6.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ap-4ffb161f.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ap-ddb28fec.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ap-00e169fa.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_b0_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_l2_ns_475': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth',
        input_size=(3, 475, 475), pool_size=(15, 15), crop_pct=0.936),
    'tf_efficientnet_l2_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth',
        input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.96),

    'tf_efficientnet_es': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), ),
    'tf_efficientnet_em': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_el': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'tf_efficientnet_cc_b0_4e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_4e-4362b6b2.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b0_8e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_8e-66184a25.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b1_8e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b1_8e-f7c79ae1.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),

    'tf_efficientnet_lite0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, interpolation='bilinear'),
    'tf_efficientnet_lite4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.920, interpolation='bilinear'),

    'tf_efficientnetv2_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s-eb54923e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m-cc09e0cd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l-d664b728.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_s_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21ft1k-d7dafa41.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21ft1k-bf41664a.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21ft1k-60127a9d.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_xl_in21ft1k': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_s_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21k-6337ad01.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21k-361418a2.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21k-91a19ec9.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_xl_in21k': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth',
        input_size=(3, 192, 192), test_input_size=(3, 224, 224), pool_size=(6, 6)),
    'tf_efficientnetv2_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b1-be6e41b0.pth',
        input_size=(3, 192, 192), test_input_size=(3, 240, 240), pool_size=(6, 6), crop_pct=0.882),
    'tf_efficientnetv2_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b2-847de54e.pth',
        input_size=(3, 208, 208), test_input_size=(3, 260, 260), pool_size=(7, 7), crop_pct=0.890),
    'tf_efficientnetv2_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b3-57773f13.pth',
        input_size=(3, 240, 240), test_input_size=(3, 300, 300), pool_size=(8, 8), crop_pct=0.904),

    'mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth'),
    'mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
    'mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth'),
    'mixnet_xl': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth'),
    'mixnet_xxl': _cfg(),

    'tf_mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
    'tf_mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
    'tf_mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),
}


class EfficientNet(nn.Module):
    """ (Generic) EfficientNet
    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
                 output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
                 se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features)
        self.act2 = act_layer(inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor
    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3,
                 stem_size=32, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels,
                 act_layer=None, norm_layer=None, se_layer=None, drop_rate=0., drop_path_rate=0.):
        super(EfficientNetFeatures, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate,
            feature_location=feature_location)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())

def _create_effnet(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = EfficientNet
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'global_pool')
        model_cls = EfficientNetFeatures
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    if features_only:
        model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model

def _gen_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else round_chs_fn(1280),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model

@register_model
def mobilenetv2_100(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.0 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_140(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.4 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_140', 1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_110d(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.1 channel, 1.2 depth multipliers"""
    model = _gen_mobilenet_v2(
        'mobilenetv2_110d', 1.1, depth_multiplier=1.2, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_120d(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.2 channel, 1.4 depth multipliers """
    model = _gen_mobilenet_v2(
        'mobilenetv2_120d', 1.2, depth_multiplier=1.4, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model

