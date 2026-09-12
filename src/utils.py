"""Shared utility functions.

中文说明：
这个模块放项目中多个脚本都会用到的通用工具函数。
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch.

    中文说明：
    固定 Python、NumPy 和 PyTorch 的随机种子，尽量提高实验可复现性。
    cudnn.benchmark=True 有利于固定输入长度时的速度；因此这里没有强制完全确定性。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
