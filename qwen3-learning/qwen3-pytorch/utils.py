"""
Qwen3 线的通用工具函数。

与 GPT-2 线的 utils.py **不共享代码**（AGENTS.md 硬约束：Qwen3 线不跨目录 import），
但接口保持一致，方便对照阅读。
"""

import os
import random
from typing import Optional

import torch


def set_seed(seed: int) -> None:
    """
    固定 Python / NumPy / PyTorch 三处随机种子，保证实验可复现。

    参数：
        seed: 随机种子
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    选设备：有 CUDA 用 CUDA，否则用 CPU（本项目 CPU 就能跑）。

    返回：
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(payload: dict, path: str) -> str:
    """
    保存 checkpoint（自动建目录）。

    参数：
        payload: 要保存的字典
        path:    保存路径
    返回：
        实际写入的路径
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location: Optional[str] = None) -> dict:
    """
    加载 checkpoint。

    参数：
        path:        路径
        map_location: torch.load 的 map_location
    返回：
        checkpoint 字典
    """
    return torch.load(path, map_location=map_location, weights_only=False)
