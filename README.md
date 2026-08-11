# CS336 (2026) — Language Models From Scratch

个人课程仓库：官方作业脚手架以**普通目录**形式纳入本仓（非 submodule），实现与笔记都只提交到本仓库，不 push 到 Stanford 官方仓。

## 结构

```text
cs336_2026/
├── assignment1-basics/      # Basics：tokenization, Transformer, train loop
├── assignment2-systems/     # (稍后) kernels, DDP, profiling
├── assignment3-scaling/     # (稍后) scaling laws
├── assignment4-data/        # (稍后) filtering, dedup
├── assignment5-alignment/   # (稍后) SFT / RL
└── notes/
    ├── assignment1/         # 作业1 书面题草稿 / 实验记录
    ├── assignment2/         # …
    └── …
```

## 环境

- 按各作业目录内的 `README.md` / `pyproject.toml` 安装依赖（推荐 `uv`）。
- 本地先跑通官方 unit tests，再上集群做训练与 benchmark。

### Assignment 1 快速开始

```bash
cd assignment1-basics
uv sync
uv run pytest
```

实现代码写在 `cs336_basics/`，并通过 `tests/adapters.py` 对接测试。Handout：`cs336_assignment1_basics.pdf`。

## 工作流

- 在对应 `assignmentN-*/` 目录中修改代码与跑测试。
- 书面题与实验记录写在 `notes/assignmentN/`。
- **所有提交都在本仓库完成**；不要把作业目录重新初始化成指向官方仓的 git 仓库后误 push。

### 同步官方脚手架更新（可选）

需要对照 Stanford 上游变更时，在仓库外临时 clone，再 diff / 手动合并：

```bash
git clone --depth 1 https://github.com/stanford-cs336/assignment1-basics.git /tmp/a1-upstream
diff -ru /tmp/a1-upstream/tests assignment1-basics/tests
# 确认后只合并你需要的文件；不要在 assignment1-basics/ 里重新 git init 并 push 到官方仓
```

日常开发把本仓库当普通单仓即可，无需 submodule 相关命令。

## 官方链接

- [课程主页 / 讲义](https://cs336.stanford.edu/)
- [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
- [assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
- [assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling)
- [assignment4-data](https://github.com/stanford-cs336/assignment4-data)
- [assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)
