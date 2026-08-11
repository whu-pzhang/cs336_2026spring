# CS336 (2026) — Language Models From Scratch

伞形仓库：每个作业是官方 assignment 的 **git submodule**，在对应目录里实现与测试。

## 结构

```text
cs336_2026/
├── assignment1-basics/      # submodule: tokenization, Transformer, train loop
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

## 拉取 / 更新官方脚手架（submodule）

首次 clone 本仓库后初始化子模块：

```bash
git submodule update --init --recursive
```

新增作业（示例）：

```bash
git submodule add https://github.com/stanford-cs336/assignment1-basics.git assignment1-basics
git submodule add https://github.com/stanford-cs336/assignment2-systems.git assignment2-systems
git submodule add https://github.com/stanford-cs336/assignment3-scaling.git assignment3-scaling
git submodule add https://github.com/stanford-cs336/assignment4-data.git assignment4-data
git submodule add https://github.com/stanford-cs336/assignment5-alignment.git assignment5-alignment
```

拉取官方脚手架更新：

```bash
# 更新某一个
git submodule update --remote assignment1-basics

# 或更新全部
git submodule update --remote --merge
```

在子模块内提交你的实现（每个作业仓库各自有独立 history）：

```bash
cd assignment1-basics
git checkout -b my-work   # 如需要
# ... 编辑 cs336_basics/、tests/adapters.py ...
git add -A && git commit -m "WIP: assignment1"
```

然后在伞形仓库记录子模块指针：

```bash
cd ..
git add assignment1-basics
git commit -m "Bump assignment1-basics submodule pointer"
```

## 官方链接

- [课程主页 / 讲义](https://cs336.stanford.edu/)
- [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
- [assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
- [assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling)
- [assignment4-data](https://github.com/stanford-cs336/assignment4-data)
- [assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)
