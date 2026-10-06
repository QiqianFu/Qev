# Qev-0.8B 训练方法

[English](training-0.8b.md) · [模型卡](model-card-0.8b.md) · [通用训练说明](training.md)

Qev-0.8B v0.1.0 从 **Qwen3.5-0.8B-Base** 直接初始化新的 LoRA、决策头和候选交互门，用 Qev-9B v0.3.0 的选项概率训练两轮。这里的“从头训练”指从预训练 Qwen 底座开始适配，不是从随机参数预训练语言模型。没有先做硬标签训练，也没有 2B 的响应蒸馏续训阶段。

## 目标与配方

教师缓存未缩放的 logits；训练时除以温度 1.563437713227029 后做 softmax。学生温度为 1，以教师分布计算交叉熵：

\[
p_T=\operatorname{softmax}(z_T/1.563437713227029),\qquad
\mathcal L=-\sum_i p_T(i)\log p_S(i).
\]

不混合硬标签损失，不添加响应匹配损失，也不乘温度平方。[蒸馏实现](../qev/distillation.py)与 4B 共用。

| 设置 | 发布版本 |
|---|---|
| 底座 | Qwen3.5-0.8B-Base，版本 `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| 适配参数 | LoRA rank 64、alpha 128；256 维、4 个注意力头、2 层决策头 |
| 候选交互 | 最后一个完整注意力层中进行兄弟分支交互 |
| 输入长度 | state 4096，question 512，candidate 256，完整路径 4096 tokens |
| 教师 | Qev-9B v0.3.0 |
| 数据 | 44,576 条单题输入，与发布的 Qev-4B 使用同一输入池和教师概率 |
| 日程 | 2 轮，2786 步，seed 17 |
| 全局批大小 | 每卡 8 × 2 卡 × 梯度累积 2 = 32 |
| 学习率 | LoRA 2e-5，决策头 1e-4，交互门 0.01 |
| 预热 | 前 100 步仅训练决策头，学习率预热 20 步 |
| 精度与执行 | BF16 主干、FP32 决策头和归约，树形批量执行 |

数据包括一般决策、科学推理、2,230 道 Principle 判断题、受控边界、网页操作和规则推理题。展开为单题并按完整输入与选项顺序去重；排除 308 道世界知识题和 3,253 道属于响应探针保留父题的问题。全部 44,576 条都通过长度检查。使用单一主池，不设收尾包，不在线插入 None 选项。

选用 seed 17。两个蒸馏种子在 JevBench 都是 169/231，普通训练对应为 162/231 和 163/231。蒸馏的概率质量更好，但并非所有评测的准确率最高。

完整训练输入池与教师缓存不公开；[Qev-train](https://huggingface.co/datasets/AustinFu/Qev-train)仍是单独发布的 2,442 条带硬标签的合成题。下面可以在自己的数据上运行相同方法，不能据此保证复现发布模型的分数。

## 在自己的数据上训练

按根 README 安装环境，然后执行：

```bash
python -m qev.prepare --input examples/train.jsonl \
  --validation examples/dev.jsonl --out data/support-0.8b

python -m qev.teacher logits \
  --teacher AustinFu/Qev-9B@v0.3.0 \
  --data data/support-0.8b --config configs/qev-0.8b-distill.json \
  --out data/teacher-logits-0.8b --device cuda

torchrun --standalone --nproc_per_node=2 -m qev.train \
  --config configs/qev-0.8b-distill.json --data data/support-0.8b \
  --out runs/qev-0.8b
```

不传 `--init-checkpoint` 就从 Qwen 底座初始化。单卡可改成 `batch_size: 1`、`accum: 32`，保留全局批大小。教师打分时不读取硬标签；缓存以温度 1 保存原始 logits，训练配置决定软化温度。预构建的规范数据在纯教师监督下允许省略标签，但必须为所有输入和选项顺序准备教师缓存。

`configs/qev-0.8b.json` 是旧的 rank-16 研究配置；本发布版使用 `configs/qev-0.8b-distill.json`。

## 微调已发布模型

在自己的标注数据上继续训练时不需要教师：

```bash
python -m qev.train --config configs/qev-0.8b-finetune.json \
  --data data/support-0.8b --out runs/qev-0.8b-support \
  --init-checkpoint AustinFu/Qev-0.8B@v0.1.0
```

这会重新建立优化器与训练日程。推理、离线加载和导出见[检查点说明](checkpoints.md)。
