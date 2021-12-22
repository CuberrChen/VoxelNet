import sys
from collections import OrderedDict

import paddle
from paddle.nn import functional as F

class Empty(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

class Sequential(paddle.nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()