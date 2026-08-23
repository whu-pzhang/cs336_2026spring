# CS336 作业进度

最后更新：2026-08-23

实现写在各 `assignmentN-*/`，书面草稿在 `notes/assignmentN/writeup.md`。本文件只记进度，不记题解。

---

## Assignment 1 — Basics

代码：`assignment1-basics/`  
书面：`notes/assignment1/writeup.md`  
推导笔记：`notes/assignment1/notes.md`

当前阶段：**提供的单测全部通过（46 passed / 2 skipped）；训练循环脚本已有，下一步在 CUDA 上跑 TinyStories 全量，并收尾书面题。**

### 实现

| 模块 | 状态 | 位置 |
|------|------|------|
| BPE 训练 `train_bpe` | 完成 | `cs336_basics/tokenizer.py` |
| `Tokenizer` encode / decode / iterable | 完成 | 同上 |
| Linear / Embedding / RMSNorm / SwiGLU / RoPE | 完成 | `cs336_basics/llm.py` |
| softmax / SDPA / MHA / TransformerBlock / LM | 完成 | 同上 |
| SiLU | 完成 | `llm.silu` → `adapters.run_silu` |
| cross-entropy | 完成 | `llm.cross_entropy` → `adapters.run_cross_entropy` |
| AdamW | 完成 | `cs336_basics/optimizer.py` → `adapters.get_adamw_cls` |
| cosine LR schedule | 完成 | `optimizer.get_lr_cosine_schedule` |
| gradient clipping | 完成 | `optimizer.gradient_clipping`（全局 L2） |
| get_batch | 完成 | `cs336_basics/training.py` → `adapters.run_get_batch` |
| checkpoint save / load | 完成 | `training.save/load_checkpoint` → 对应 adapters |
| 训练循环 | 完成 | `experiments/train.py`（超参 CLI + memmap + checkpoint + 日志） |

### 单测

**全套通过：46 passed / 2 skipped（skipped 为 Linux 专属的 tokenizer 内存测试，macOS 正常跳过）。**

包括 `test_get_batch`、`test_checkpointing`；此前挂的 tokenizer 与 tiktoken 对齐 / special-token 边界测试也已全部通过。

### 书面题

| 题 | 状态 |
|----|------|
| unicode1 / unicode2 | 草稿已写 |
| transformer_accounting | 草稿已写 |
| learning_rate_tuning | 未写（下一道可做的问答，无需数据） |
| adamwAccounting | 草稿部分写了，仍有待填（最大 batch / AdamW FLOPs / 训练天数） |
| train_bpe_tinystories / train_bpe_expts_owt | 产物已齐（vocab/merges/图），writeup 未写 |
| tokenizer_experiments | 脚本已有，writeup 未写 |
| 第 5–7 节训练 / 消融实验 | 未写 |

### 实验与交付

- [x] 下载 TinyStories / OWT（2026-08-23 经 hf-mirror.com 下载）
- [x] 在真实语料上训 BPE，导出 vocab / merges（`experiments/artifacts/`）
- [x] tokenize 成 `uint16` `.npy`（`data/tokenized/`，gitignored）
- [x] 训练循环脚本（讲义 5.3；冒烟 50 step loss 下降）
- [ ] 端到端训练 Transformer LM（CUDA，TinyStories 40k step）
- [ ] 实验记录、生成样例、`writeup.pdf`

### 建议下一步

1. CUDA 上跑 TinyStories 全量（valid loss ≤ 1.45）
2. 书面：`learning_rate_tuning` → 补完 `adamw_accounting` → 根据已有 BPE 产物写实验题
3. 消融实验 → 写报告

---

## Assignment 2 — Systems

未开始。

## Assignment 3 — Scaling

未开始。

## Assignment 4 — Data

未开始。

## Assignment 5 — Alignment

未开始。
