# -*- coding: utf-8 -*-
"""
train_ppo.py —— PPO 训练的标准入口（thin wrapper）。

README 里给的命令是 `python train_ppo.py`，但完整实现写在
`train_ppo_standard_adapted.py`（带 4 个角色拆分 + token-level value +
KL reward shaping + GAE 等标准化改造）。本文件仅做 import + 转发，
不写任何训练逻辑，方便阅读时一眼看出"入口在哪"。
"""

# 直接 import 标准版所有符号；它本身是顶层线性脚本，import 即触发主流程。
# 如果用户希望"只 import 不跑"，可加 if __name__ == "__main__" 后改用
# `from train_ppo_standard_adapted import <符号>` 形式调用。
from train_ppo_standard_adapted import *  # noqa: F401, F403
