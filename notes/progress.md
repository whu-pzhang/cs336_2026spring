# CS336 作业进度

最后更新：2026-08-19

实现写在各 `assignmentN-*/`，书面草稿在 `notes/assignmentN/writeup.md`。本文件只记进度，不记题解。

---

## Assignment 1 — Basics

代码：`assignment1-basics/`  
书面：`notes/assignment1/writeup.md`  
推导笔记：`notes/assignment1/notes.md`

当前阶段：**模型 + cross-entropy 已齐；训练栈（AdamW / LR / clip / batch / checkpoint）未开始。**

### 实现

| 模块 | 状态 | 位置 |
|------|------|------|
| BPE 训练 `train_bpe` | 完成 | `cs336_basics/tokenizer.py` |
| `Tokenizer` encode / decode / iterable | 完成 | 同上 |
| Linear / Embedding / RMSNorm / SwiGLU / RoPE | 完成 | `cs336_basics/llm.py` |
| softmax / SDPA / MHA / TransformerBlock / LM | 完成 | 同上 |
| SiLU | 完成 | `llm.silu` → `adapters.run_silu` |
| cross-entropy | 完成 | `llm.cross_entropy` → `adapters.run_cross_entropy` |
| gradient clipping | 未做 | `adapters.run_gradient_clipping` |
| AdamW | 未做 | `adapters.get_adamw_cls`（`optimizer.py` 仍空） |
| cosine LR schedule | 未做 | `adapters.run_get_lr_cosine_schedule` |
| get_batch | 未做 | `adapters.run_get_batch` |
| checkpoint save / load | 未做 | `adapters.run_save/load_checkpoint` |
| 训练循环 | 未做 | — |

### 单测

已接线：`test_train_bpe`、`test_tokenizer`、`test_model`、`test_softmax`、`test_cross_entropy`。

仍会 `NotImplementedError`：

- `test_gradient_clipping`
- `test_adamw` / `test_get_lr_cosine_schedule`
- `test_get_batch`
- `test_checkpointing`

### 书面题

| 题 | 状态 |
|----|------|
| unicode1 / unicode2 | 草稿已写 |
| transformer_accounting | 草稿已写 |
| learning_rate_tuning | 未写（下一道可做的问答，无需数据） |
| adamwAccounting | 未写（实现 AdamW 前后） |
| train_bpe_tinystories / train_bpe_expts_owt | 未写（需下数据并训 BPE） |
| tokenizer_experiments | 未写（需已训好的 tokenizer） |
| 第 5–7 节训练 / 消融实验 | 未写 |

### 实验与交付

- [ ] 下载 TinyStories / OWT（尚无 `data/`）
- [ ] 在真实语料上训 BPE，导出 vocab / merges
- [ ] 端到端训练 Transformer LM
- [ ] 实验记录、生成样例、`writeup.pdf`

### 建议下一步

1. 书面：`learning_rate_tuning`（讲义 SGD 小例子，换 1e1 / 1e2 / 1e3）
2. 实现：AdamW → LR schedule → clip → get_batch → checkpoint
3. 书面：`adamwAccounting`
4. 下数据 → 训 tokenizer → 训练循环 → 写报告

---

## Assignment 2 — Systems

未开始。

## Assignment 3 — Scaling

未开始。

## Assignment 4 — Data

未开始。

## Assignment 5 — Alignment

未开始。
