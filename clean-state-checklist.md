# Clean State Checklist — llm-learning

> 收班清单。**6 项全勾才算这个 session 干净结束**（见 `AGENTS.md` §8 第 5 步）。
> 任一项没勾上，不要关闭 session，先把它解决或写进 `session-handoff.md`。

---

- [ ] **标准启动路径仍可用**：`bash init.sh` 跑完，退出码为 0（`feat-001` 未解决时 L2 SKIP 可接受，但要在 progress 里写明）
- [ ] **标准验证路径仍成立**：本次改动涉及的层级（L0/L1/L2/L3）都实际跑过，不是靠记忆
- [ ] **进度已记录**：`claude-progress.md` 的 Current Verified State 已更新，Session Log 已追加本次条目
- [ ] **Feature 状态反映真实情况**：`feature_list.json` 里的 `status` 和 `evidence` 没有浮夸；没做完的就是没做完
- [ ] **没有没文档化的半成品**：改动过的训练/评估代码要么跑通，要么在 `feature_list.json` 的 `notes` 里写明坏在哪
- [ ] **下一个 session 能直接继续**：从干净 clone 执行 `bash init.sh` 不需要任何手工修补，接着读 `feature_list.json` 就能开工

---

## 附：本项目特有的额外自查

- [ ] 本次改 `config.py` 的话，`notes` 里写明了它影响哪几章笔记
- [ ] 本次新增/修改笔记的话，至少抽样核对 3 处数字或代码片段与源码一致
- [ ] 工作区没有待提交的大文件或权重（`git status` 干净，`checkpoints/`、`docs/` 仍被忽略）
