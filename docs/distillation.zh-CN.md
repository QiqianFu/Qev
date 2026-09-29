# 用 Qev-9B 教师训练 Qev-2B

Qev-2B 沿用 Qev-9B 的决策结构，将底座换为 Qwen3.5-2B-Base。训练分为两阶段：先用教师的选项概率做交叉熵训练，再加入程序化文本编辑、原题回放和内部表示响应蒸馏。

当前选择的模型不使用候选预览，MMLU-Pro 为 38.70%，JevBench 公开题为 172/231。[评测文档](evaluation.md)按相同题目子集对比两个模型尺寸。[English](distillation.md)。

## 第一阶段 学习教师概率

从 Qwen3.5-2B-Base 开始训练 rank-64 LoRA、256 维两层决策头和候选交互参数，教师为 Qev-9B。

教师为每道题的完整候选集合提供概率分布，学生以该分布为目标计算交叉熵。所选模型的教师权重为 1，训练时没有额外加入原始标签损失。通用训练器也支持降低 `training.distillation.weight`，将原目标与教师概率混合。

缓存覆盖学生实际使用的输入，包括“以上都不是”选项的插入或删除。教师只读取上下文、问题和选项；标签、训练目标不会进入教师输入。某个输入缺少教师结果时，训练会明确报错。

```bash
python -m qev.teacher logits \
  --teacher AustinFu/Qev-9B \
  --data data/decisions --config configs/qev-2b-distill.json \
  --out data/teacher-logits --device cuda

torchrun --standalone --nproc_per_node=2 -m qev.train \
  --config configs/qev-2b-distill.json \
  --data data/decisions --out runs/qev-2b-probabilities
```

配置使用所选模型的训练计划，要求数据含 `train` 与 `late_train` 分区。如果使用 `qev.prepare` 整理自己的数据、只有 `train` 分区，删除配置中的 `late_split`、`late_fraction` 和 `late_repeats`。将 `training.distillation.cache` 设为第一条命令生成的目录。

原训练使用主分区 34,546 条、补充分区 1,783 条；补充数据在训练后半程混入并重复三次。总计两轮，全局 batch 为 32。完整研究数据没有随代码分发；换用其他数据后，训练步数和结果也会变化。

## 准备文本编辑和教师响应

程序按固定随机种子改动数字、反转比较条件，或删除一条背景观察。问题和选项保持不变，选择改动时不读取标签。程序生成的是改写候选，教师置信度只是筛选依据，不能保证每次编辑都保持合理语义。

所选配方要求原题和改写题的教师最高选项概率均大于 0.8。题对按父题分组划分训练与诊断部分，并移除跨分区的相同输入。原题回放是把未改写的训练题重新混入训练；回放题也排除诊断题的父组。此阶段排除了世界知识子集。

```bash
python -m qev.teacher responses \
  --teacher AustinFu/Qev-9B \
  --data data/decisions --config configs/qev-2b-distill.json \
  --out data/teacher-responses --device cuda
```

输出包含 `replay.jsonl`、`pairs.jsonl`、`probe.jsonl` 和数据说明。同一次教师前向同时产生概率与内部表示目标。原续训使用 37,835 道回放题和 15,806 对训练题；诊断题不参与梯度或归一化尺度计算。

准备命令会按相同编辑算子与置信度规则，为你提供的数据生成新目标。精确重复已发布模型的训练，还需要原来选定的数据及教师目标；这些数据没有随代码分发。

## 第二阶段 混合交叉熵和表示响应损失

在决策头中，依次取出问题向量和各候选向量，并分别归一化。原题与改写题对应的矩阵为 $H$ 和 $H'$，定义响应：

$$R=(H'-H)H^\top.$$

它描述改写带来的变化与原问题、原选项之间的关系。教师与学生分别计算自己的响应矩阵，不要求两者的神经元坐标一致。

每步处理 $B=32$ 个输入，损失为：

$$L=\frac{1}{B}\left[\sum_x \mathrm{CE}(\mathrm{softmax}(z_T(x)/T_T),\mathrm{softmax}(z_S(x)))+\frac{2\lambda}{s}\sum_{(x,x')}\mathrm{MSE}(R_S,R_T)\right].$$

每对题包含两个输入，因此响应项乘 2。$s$ 是训练题对的教师响应平方均值，并设置一个很小的下限；所选数据约为 0.00671145。学生温度为 1，教师温度为 1.563437713227029，响应权重 $\lambda=0.1$。

```bash
python -m qev.distill \
  --checkpoint runs/qev-2b-probabilities/step-002327 \
  --data data/teacher-responses --config configs/qev-2b-response.json \
  --out runs/qev-2b-response --device cuda
```

检查点填写第一阶段实际保存的最后一步。所选配方在单卡上续训 800 步，每步混合 24 道原题和 4 对原题／改写题。两个池分别洗牌、循环取样；microbatch 最多容纳 16 个输入，同一题对始终放在一起。

| 设置 | 所选配方 |
|---|---|
| LoRA／决策头／交互参数学习率 | 0.00001／0.00005／0.005 |
| 预热 | 20 步 |
| 学习率计划 | 余弦下降，最低为初始值的 10% |
| 权重衰减 | 0.01 |
| 梯度裁剪 | 1.0 |
| 随机种子 | 17 |
| 保存位置 | 400 与 800 步 |

中断后，在原命令中增加 `--resume runs/qev-2b-response/step-000400`。模型、优化器、随机状态与采样位置都会恢复；数据和配置应保持不变。

## 导出和使用

```bash
python -m qev.export \
  --checkpoint runs/qev-2b-response/step-000800 \
  --out checkpoints/qev-2b

python -m qev.predict --checkpoint checkpoints/qev-2b \
  --input examples/requests.jsonl --out runs/qev-2b-predictions.jsonl --device cuda
```

导出后使用标准 Qev 接口推理，不再需要加载教师。[模型卡](model-card-2b.md) · [数据格式](data.md) · [许可证](../THIRD_PARTY_NOTICES.md)。
