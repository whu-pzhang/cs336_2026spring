# CS336 作业进度

最后更新：2026-08-21

实现写在各 `assignmentN-*/`，书面草稿在 `notes/assignmentN/writeup.md`。本文件只记进度，不记题解。

---

## Assignment 1 — Basics

代码：`assignment1-basics/`  
书面：`notes/assignment1/writeup.md`  
推导笔记：`notes/assignment1/notes.md`

当前阶段：**优化器栈（AdamW / cosine LR / gradient clipping）已齐；还差 get_batch、checkpoint 与训练循环。**

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
| get_batch | 未做 | `adapters.run_get_batch` |
| checkpoint save / load | 未做 | `adapters.run_save/load_checkpoint` |
| 训练循环 | 未做 | — |

### 单测

已通过相关：`test_train_bpe`、`test_model`、`test_softmax`、`test_cross_entropy`、`test_adamw`、`test_get_lr_cosine_schedule`、`test_gradient_clipping`；tokenizer 部分 roundtrip 通过，部分与 tiktoken 对齐 / special-token 边界仍挂。

仍会 `NotImplementedError`：

- `test_get_batch`
- `test_checkpointing`

### 书面题

| 题 | 状态 |
|----|------|
| unicode1 / unicode2 | 草稿已写 |
| transformer_accounting | 草稿已写 |
| learning_rate_tuning | 未写（下一道可做的问答，无需数据） |
| adamwAccounting | 草稿部分写了，仍有待填（最大 batch / AdamW FLOPs / 训练天数） |
| train_bpe_tinystories / train_bpe_expts_owt | 未写（需下数据并训 BPE） |
| tokenizer_experiments | 未写（需已训好的 tokenizer） |
| 第 5–7 节训练 / 消融实验 | 未写 |

### 实验与交付

- [ ] 下载 TinyStories / OWT（尚无 `data/`）
- [ ] 在真实语料上训 BPE，导出 vocab / merges
- [ ] 端到端训练 Transformer LM
- [ ] 实验记录、生成样例、`writeup.pdf`

### 建议下一步

1. 实现：get_batch → checkpoint save/load → 训练循环
2. 书面：补完 `adamwAccounting` 待填；或先做 `learning_rate_tuning`
3. 下数据 → 训 tokenizer → 端到端训练 → 写报告

---

## Assignment 2 — Systems

未开始。

## Assignment 3 — Scaling

未开始。

## Assignment 4 — Data

未开始。

## Assignment 5 — Alignment

未开始。
