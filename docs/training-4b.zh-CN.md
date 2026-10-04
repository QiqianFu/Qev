# Qev-4B 训练方法

[English](training-4b.md) · [模型卡](model-card-4b.md) · [通用训练指南](training.md)

Qev-4B v0.1.0 **从 Qwen3.5-4B-Base 直接初始化**，新建 rank-64 LoRA、256 维两层决策头和末层候选交互门控，再用 9B 教师的选项概率进行单阶段训练。它没有继承先前蒸馏过的 4B 权重，也没有后续内部表示响应训练；从 base 初始化与使用教师概率监督是两个不同的设置。

## 训练目标

教师为给定输入的每个候选打分，先缓存未经温度缩放的 logits，训练时计算：

\[
p_T(i\mid x)=\operatorname{softmax}(z_T(x)/1.563437713227029)_i,
\qquad \mathcal L=-\sum_i p_T(i\mid x)\log p_S(i\mid x).
\]

学生温度为 1，不混入硬标签、不加响应损失，也不额外乘温度平方。教师请求去掉标签和目标概率；缓存按上下文、指令、题型、候选 ID／文字／顺序绑定。任何输入缺少教师目标都会中止训练。[实现](../qev/distillation.py)。

## 发布模型的实际配方

| 设置 | Qev-4B v0.1.0 |
|---|---|
| 底座版本 | `710fd005d44d55ee27b7ad5147e318e546efdbfe` |
| 参数适配 | LoRA rank 64、alpha 128；256 维、4 头、2 层决策头 |
| 候选交互 | 最后一个全注意力层中的完整同题候选交互 |
| 输入上限 | state 4,096；question 512；candidate 256；完整路径 4,096 token |
| 训练数据 | 44,576 条单题输入，硬标签已移除 |
| 训练进度 | 两轮，共 2,786 步，seed 17 |
| 批量 | 每卡 8 条 × 4 卡，全局 batch 32 |
| 学习率 | LoRA 2e-5；决策头 1e-4；交互门控 0.01 |
| 预热 | 前 100 步只训练决策头；学习率预热 20 步 |
| 计算方式 | BF16 主干、FP32 决策头与读出归约，树形合批训练 |
| 采样 | 单一主集，没有收尾分区、在线 None 插入或文本改写 |

输入池包含通用决策、科学与推理题、2,230 道 Principle 判断、受控边界题、网页动作和补充规则／推理题。先展开为每条一题，按完整输入与候选顺序去重；排除 308 道世界知识题，以及留出响应探针父组对应的 3,253 道题。剩余 44,576 道全部通过编码准入。

实际教师现已发布为 [Qev-9B v0.3.0](https://huggingface.co/AustinFu/Qev-9B/tree/v0.3.0)，是加入网页与规则数据、训练到 2,658 步的检查点，训练张量与这份 4B 权重使用的教师一致。完整 4B 输入池和教师缓存未分发。下载的 4B 权重保留实际训练结果；换用其他教师或数据重跑方法，不等于复现相同成绩。

## 在自己的数据上训练

按根 README 安装 Qev。下面先准备标注数据，采集教师输出时会去掉标签：

```bash
python -m qev.prepare --input examples/train.jsonl \
  --validation examples/dev.jsonl --out data/support-4b

python -m qev.teacher logits \
  --teacher AustinFu/Qev-9B@v0.3.0 \
  --data data/support-4b --config configs/qev-4b.json \
  --out data/teacher-logits-4b --device cuda

torchrun --standalone --nproc_per_node=4 -m qev.train \
  --config configs/qev-4b.json --data data/support-4b \
  --out runs/qev-4b
```

这是使用原始 9B 教师和少量示例数据的运行示例。实际任务应选择合适的教师与更多训练数据；采集 logits 时仍受教师自身输入长度限制。省略 `--init-checkpoint` 就从 Qwen 底座开始。缓存保持温度 1，在训练端通过 `training.distillation.temperature` 缩放。改用单卡时，可设置 `batch_size: 1`、`accum: 32` 保持全局 batch 不变。

已有标准 JSONL 数据可在 `training.distillation.weight` 为 1 时省略 `label` 和 `target`，manifest 中训练分区的 `role` 必须为 `train`。混合教师概率与硬标签时仍须提供原目标。`qev.prepare` 继续用于准备带标签的数据。

## 微调已发布的 4B

使用不需要教师缓存的监督微调配置：

```bash
python -m qev.train --config configs/qev-4b-finetune.json \
  --data data/support-4b --out runs/qev-4b-support \
  --init-checkpoint AustinFu/Qev-4B@v0.1.0
```

这会新建优化器与学习率计划。单卡批量仅是运行配置示例，不是最低显存承诺。推理、本地底座、导出与断点恢复见[检查点指南](checkpoints.md)。
