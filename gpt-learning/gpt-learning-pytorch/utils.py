"""
通用工具集合。
- set_seed：固定 Python / NumPy / PyTorch 随机源，确保可复现
- get_device：自动选择 CPU / GPU
- save_checkpoint / load_checkpoint：统一的 checkpoint 读写接口
- 计划在 P2 / P3 阶段继续补（get_lr、plot_losses、AverageMeter 等）

P0 阶段只放"能 import"的最小集；所有函数均可立即调用，行为符合函数 docstring 描述。
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42):
    """固定 Python / NumPy / PyTorch 三处随机源。

    参数：
        seed: 随机种子（默认 42）
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA 仅在可用时设置；CPU 学习项目不影响正确性
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """自动选择 CPU / GPU 设备。

    返回：'cuda' 或 'cpu' 字符串。
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(obj, path: str):
    """统一 checkpoint 写入接口。自动创建父目录。

    参数：
        obj: 待保存对象（PyTorch 模型 state_dict / 完整 dict）
        path: 目标文件路径（如 'checkpoints/pretrain.pt'）
    """
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    print(f"[checkpoint] saved → {path}")


def load_checkpoint(path: str):
    """加载 checkpoint。

    参数：
        path: 文件路径

    返回：torch.load 的原始对象。
    """
    import torch
    return torch.load(path, map_location="cpu")
