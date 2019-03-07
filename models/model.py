from pretrainedmodels.models import bninception
from torch import nn


def get_bninception():
    model = bninception(pretrained='imagenet')
    model.global_pool = nn.AdaptiveAvgPool2d(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)