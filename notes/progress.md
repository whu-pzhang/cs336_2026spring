# CS336 作业进度

最后更新：2026-08-26

实现写在各 `assignmentN-*/`，书面草稿在 `notes/assignmentN/writeup.md`。本文件只记进度，不记题解。

---

## Assignment 1 — Basics

代码：`assignment1-basics/`  
书面：`notes/assignment1/writeup.md`  
推导笔记：`notes/assignment1/notes.md`

当前阶段：**核心实现测试通过（47 passed / 1 expected xfail）；Tokenizer、书面 accounting 和 2.7 实验已完成，仍需完成解码、端到端训练和第 7 节实验。**

### 实现

| 模块 | 状态 | 位置 |
|------|------|------|
| BPE 训练 `train_bpe` | 完成 | `cs336_basics/tokenizer.py` |
| `Tokenizer` encode / decode / iterable | 完成 | 同上 |
| BPE 训练脚本与 artifact 导出 | 完成 | `experiments/train_tokenizers.py` |
| tokenizer experiments | 完成（2.7(d) 需用新词表重编码） | `experiments/tokenizer_experiments.py` |
| 并行数据集 tokenization | 脚本完成；现有 `.npy` 为旧词表 | `experiments/tokenize_datasets.py` |
| Linear / Embedding / RMSNorm / SwiGLU / RoPE | 完成 | `cs336_basics/llm.py` |
| softmax / SDPA / MHA / TransformerBlock / LM | 完成 | 同上 |
| SiLU | 完成 | `llm.silu` → `adapters.run_silu` |
| cross-entropy | 完成 | `llm.cross_entropy` → `adapters.run_cross_entropy` |
| AdamW | 完成 | `cs336_basics/optimizer.py` → `adapters.get_adamw_cls` |
| cosine LR schedule | 完成 | `optimizer.get_lr_cosine_schedule` |
| gradient clipping | 完成 | `optimizer.gradient_clipping`（全局 L2） |
| get_batch | 完成 | `cs336_basics/training.py` → `adapters.run_get_batch` |
| checkpoint save / load | 完成 | `training.save/load_checkpoint` → 对应 adapters |
| 训练循环（training_together） | 完成；已验证训练、验证、JSONL 日志、checkpoint 和 resume | `experiments/train.py` |

### 单测

**全套通过：47 passed / 1 expected xfail。** xfail 是题目明确预期的
`Tokenizer.encode` 1MB 内存限制测试；`encode_iterable` 内存测试已通过。

包括 `test_get_batch`、`test_checkpointing`；此前挂的 tokenizer 与 tiktoken 对齐 / special-token 边界测试也已全部通过。

### 书面题

| 题 | 状态 |
|----|------|
| unicode1 / unicode2 | 已完成；补充编码长度和非法 UTF-8 示例 |
| transformer_accounting | 已完成；补充参数量、FLOPs 表格和长上下文分析 |
| learning_rate_tuning | 已写；脚本和 JSON 结果已生成 |
| adamw_accounting | 已完成；显存、batch size、AdamW FLOPs、H100 训练时间已补齐 |
| train_bpe_tinystories / train_bpe_expts_owt | 已写；当前基于 50M/5M 子集 |
| tokenizer_experiments | 已写；2.7(d) 的新词表数组仍待重生成 |
| 第 5–7 节训练 / 消融实验 | 未写 |

### 实验与交付

- [x] 下载 TinyStories / OWT（2026-08-23 经 hf-mirror.com 下载）
- [x] 在训练子集上训 10K/32K BPE，导出 vocab / merges（`experiments/artifacts/`）
- [~] `tokenize_datasets.py` 支持 8 worker；已有 `uint16` `.npy` 仍由旧 32K 词表生成
- [x] 完成 2.7 tokenizer experiment（压缩率、吞吐量、Pile 估算、histogram）
- [x] `training_together` 训练循环（超参 CLI、memmap、验证、JSONL 指标、checkpoint、resume；CPU 冒烟通过）
- [ ] 实现 decoder（temperature / top-p）并生成至少 256 token 文本
- [x] AdamW accounting（显存、batch size、AdamW FLOPs、H100 训练时间）
- [ ] 端到端训练 Transformer LM（CUDA，TinyStories 40k step）
- [ ] 学习率 sweep、batch size 实验、架构消融、OWT 主实验、leaderboard
- [ ] 实验记录、生成样例、`writeup.pdf`

### 建议下一步

1. 用新 10K/32K artifact 重生成 TinyStories/OWT train/valid `.npy`
2. 实现 decoder 并生成至少 256 token 的样例
3. CUDA 上跑 TinyStories，完成 learning-rate / batch-size sweep 和验证曲线
4. 完成架构消融、OWT 主实验、实验日志和最终 `writeup.pdf`

---

## Assignment 2 — Systems

未开始。

## Assignment 3 — Scaling

未开始。

## Assignment 4 — Data

未开始。

## Assignment 5 — Alignment

未开始。
