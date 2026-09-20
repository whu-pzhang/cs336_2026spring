# CS336 作业进度

最后更新：2026-09-20

实现写在各 `assignmentN-*/`，书面草稿在 `notes/assignmentN/writeup.md`。本文件只记进度，不记题解。

---

## Assignment 1 — Basics

代码：`assignment1-basics/`  
书面：`notes/assignment1/writeup.md`  
推导笔记：`notes/assignment1/notes.md`

当前阶段：**作业 1 到此结束。** 必做实现、实验和书面草稿已齐；7.5 实验已跑完。非 Stanford 学生，不上交课程 / leaderboard。

### 实现


| 模块                                                 | 状态                                                     | 位置                                                   |
| ---------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------ |
| BPE 训练 `train_bpe`                                 | 完成                                                     | `cs336_basics/tokenizer.py`                            |
| `Tokenizer` encode / decode / iterable               | 完成                                                     | 同上                                                   |
| BPE 训练脚本与 artifact 导出                         | 完成                                                     | `experiments/train_tokenizers.py`                      |
| tokenizer experiments                                | 完成                                                     | `experiments/tokenizer_experiments.py`                 |
| 并行数据集 tokenization                              | 完成；10K/32K 全量 train/valid `.npy` 已生成             | `experiments/tokenize_datasets.py` → `data/tokenized/` |
| Linear / Embedding / RMSNorm / SwiGLU / RoPE         | 完成                                                     | `cs336_basics/llm.py`                                  |
| softmax / SDPA / MHA / TransformerBlock / LM         | 完成                                                     | 同上                                                   |
| SiLU                                                 | 完成                                                     | `llm.silu` → `adapters.run_silu`                       |
| cross-entropy                                        | 完成                                                     | `llm.cross_entropy` → `adapters.run_cross_entropy`     |
| AdamW                                                | 完成                                                     | `cs336_basics/optimizer.py` → `adapters.get_adamw_cls` |
| cosine LR schedule                                   | 完成                                                     | `optimizer.get_lr_cosine_schedule`                     |
| gradient clipping                                    | 完成                                                     | `optimizer.gradient_clipping`（全局 L2）               |
| get_batch                                            | 完成                                                     | `cs336_basics/training.py` → `adapters.run_get_batch`  |
| checkpoint save / load                               | 完成                                                     | `training.save/load_checkpoint` → 对应 adapters        |
| 训练循环（training_together）                        | 完成；已验证训练、验证、JSONL 日志、checkpoint 和 resume | `experiments/train.py` + `experiments/trainer/`        |
| 架构消融开关（no RMSNorm / post-norm / NoPE / SiLU） | 完成；`--ablation` 接入训练脚本                          | `cs336_basics/llm.py`、`experiments/train.py`          |
| Muon / WSD schedule                                  | 完成；作业 cosine helper 未改                            | `cs336_basics/optimizer.py`、`experiments/trainer/`    |
| leaderboard 模型开关（QK-Norm / tying / zero-init / fused / compile） | 完成                                          | `cs336_basics/llm.py`、`experiments/trainer/config.py` |




### 单测

**全套通过：47 passed / 1 expected xfail。** xfail 是题目明确预期的
`Tokenizer.encode` 1MB 内存限制测试；`encode_iterable` 内存测试已通过。

包括 `test_get_batch`、`test_checkpointing`；此前挂的 tokenizer 与 tiktoken 对齐 / special-token 边界测试也已全部通过。

### 书面题


| 题                                          | 状态                                                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| unicode1 / unicode2                         | 已完成；补充编码长度和非法 UTF-8 示例                                                           |
| transformer_accounting                      | 已完成；补充参数量、FLOPs 表格和长上下文分析                                                    |
| learning_rate_tuning                        | 已写；脚本和 JSON 结果已生成                                                                    |
| adamw_accounting                            | 已完成；显存、batch size、AdamW FLOPs、H100 训练时间已补齐                                      |
| train_bpe_tinystories / train_bpe_expts_owt | 已写；当前基于 50M/5M 子集                                                                      |
| tokenizer_experiments                       | 已写；2.7(d) 已用 10K/32K 词表重编码                                                            |
| 第 5–7 节训练 / 消融 / OWT                  | TinyStories 主实验、LR / batch、生成、7.3 消融、7.4 OWT 已记 |
| 7.5 leaderboard（可选）                     | 实验已完成；不上交课程榜 / 不写入 writeup（非 Stanford）     |




### 实验与交付

- [x] 下载 TinyStories / OWT（2026-08-23 经 hf-mirror.com 下载）
- [x] 在训练子集上训 10K/32K BPE，导出 vocab / merges（`experiments/artifacts/`）
- [x] 用 TinyStories 10K / OWT 32K 词表重编码 train/valid（8 worker，`data/tokenized/`；meta 见 `data/tokenized/meta.json`）
- [x] 完成 2.7 tokenizer experiment（压缩率、吞吐量、Pile 估算、histogram）
- [x] `training_together` 训练循环（超参 CLI、memmap、验证、JSONL 指标、checkpoint、resume；CPU 冒烟通过）
- [x] 用 TinyStories checkpoint 生成至少 256 token 文本（`experiments/decoding.py`，temperature=0.9，top-p=0.9）
- [x] AdamW accounting（显存、batch size、AdamW FLOPs、H100 训练时间）
- [x] 端到端训练 Transformer LM（CUDA，TinyStories 40k step；`experiments/artifacts/tinystories_lm/ckpt.pt`）
- [x] TinyStories 学习率 sweep（`1e-4` / `3e-4` / `1e-3` / `3e-3`；1e-3 最好）
- [x] TinyStories batch size sweep（16 / 32 / 64 / 128，对齐 token；32 与 128 接近，16 更差）
- [x] 架构消融（NoPE / SiLU / post-norm / 去 RMSNorm；`1e-3` 去 norm 发散，`1e-4` 最终 valid 2.188）
- [x] OWT 主实验（同架构 40k；最终 valid 4.056，约 45.5 min；生成样例已记）
- [x] 训练脚本重构子项目 1：拆成 `experiments/trainer/` 包（config / builder / metrics / loop），
      config 成为可序列化对象并落盘 `config.json`；行为门禁（100 步 loss 逐位相同）通过
- [x] leaderboard 10k 消融扫（`experiments/artifacts/iters10k/`；QK-Norm / tying / Muon / zero-init / L6 / fused+compile）
- [x] leaderboard 50k 长跑（可选 7.5；5090 D，batch 64，fused+compile；WSD valid **3.496** @ 78.9 min，cosine 3.513 @ 78.7 min；均 < 5.0。正式榜是 45 min B200）
      最佳 checkpoint：`experiments/artifacts/leaderboard/final/`（WSD）。配置：`experiments/configs/owt_ctx512_final_bf16.json`
- [x] 课程提交跳过（非 Stanford：不上榜、不排 `writeup.pdf`、不打包）



### 建议下一步

Assignment 1 到此结束。下一份是 Assignment 2 — Systems。

---



## Assignment 2 — Systems

未开始。

## Assignment 3 — Scaling

未开始。

## Assignment 4 — Data

未开始。

## Assignment 5 — Alignment

未开始。
