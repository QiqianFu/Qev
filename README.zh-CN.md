<div align="center">
  <img src="assets/banner.svg" alt="Qev — 基于 Qwen 的决策模型" width="100%">
  <p><strong>共享上下文，独立候选分支，统一决策头。</strong></p>
  <p><a href="README.md">English</a> · <a href="docs/decision-head.html#set-head">交互式决策头图解</a> · <a href="docs/evaluation.md">评测协议</a> · <a href="docs/training.md">训练方法</a></p>
</div>

Qev 从你提供的候选答案中做决策。输入共享上下文、问题和候选，输出每个问题的概率分布；同一个模型处理 **Choice 选择、Noul 是非判断、Score 序数评分**。

**Qev-9B** 基于 **Qwen3.5-9B-Base**，使用 rank-64 LoRA、候选分支读出与 256 维集合决策头。选定检查点在 JevBench 的 231 道公开题上答对 **188 题（81.39%）**；固定版本 Kev-9B 为 175 题（75.76%），托管模型 Jev 的参考成绩为 198 题（85.71%）。[详细口径与来源](docs/evaluation.md)

本仓库包含模型、训练和评测代码、可移植检查点工具、测试与示例。研究阶段的名称是 BranchKev，旧数据和检查点格式仍可读取。目前尚未分配公开权重下载地址；可按下文导出本地检查点，完整研究训练集也未随代码分发。

## 快速开始

使用 Python 3.12，先安装适合硬件的 PyTorch 2.8.0。在本仓库根目录执行：

```bash
python -m pip install -e '.[test]'
# 无需下载模型，在CPU验证准备数据、训练、续训和推理。
python scripts/smoke.py --out runs/smoke
```

短试跑使用随机初始化的小型 Qwen，只验证软件流程，不代表 Qev-9B 的效果。

将导出的正式检查点放在 `checkpoints/qev-9b` 后运行：

```bash
python -m qev.predict \
  --checkpoint checkpoints/qev-9b \
  --input examples/requests.jsonl --out runs/predictions.jsonl \
  --device cuda --weights-dtype checkpoint
```

输出文件必须使用新名字。默认使用共享前缀缓存；复现正式评测的执行路径时加 `--reference`。也支持发布后以 `owner/repository@commit` 加载固定版本的Hub检查点。[检查点导出与加载](docs/checkpoints.md)

Python 接口：

```python
from qev import Qev

model = Qev.from_pretrained("checkpoints/qev-9b", device="cuda")
answers = model.predict({
    "state": "同一订单被扣款两次，请立即处理。",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "应该由哪个团队处理？",
            "criteria": {"billing": "账单和退款", "shipping": "物流配送"},
        },
        "urgent": {"type": "noul", "instructions": "是否明确要求立即处理？"},
        "priority": {
            "type": "score", "instructions": "评价请求的响应时效。",
            "criteria": ["未提出紧迫要求", "尽快", "立即"],
        },
    },
})
print(answers["department"]["probabilities"])
print(answers["urgent"]["noul"])       # P(true)
print(answers["priority"]["score"])   # 从0开始的等级期望
```

所有答案保留 `prediction` 和 `probabilities`，并按类型附带 `choice`、`noul` 或 `score`。[输入输出约定](docs/data.md)

## 模型怎样作出决策

<img src="assets/architecture.svg" alt="消息与账单、物流、账户三个选项结合，形成选项摘要向量，再比较并输出选项概率" width="100%">

图中的 **Option 就是候选答案／选项**，例子里分别是账单（Billing）、物流（Shipping）和账户（Account）团队。`e₁`、`e₂`、`e₃` 是模型结合上下文为这三个选项计算的摘要向量，`h_q` 是消息与问题的摘要向量；向量就是一组编码语义的数字。

输入组织为 **state → question → candidate**。每个候选有自己的因果分支和末尾读出标记；问题标记总结上下文与问题，候选标记总结自己的分支。正式模型的最后一层还允许候选读出通过可学习门控读取兄弟候选。

集合决策头接收的是 **1个问题向量和K个候选向量**。这里的K+1个位置已经是主干汇总后的向量：

1. **分别投影。** 问题与候选共用 LayerNorm 和线性投影，从4096维变为256维；问题另加一个可学习角色向量。这一步按向量通道计算，没有混合候选。
2. **集合内交换信息。** 堆成 `[1, K+1, 256]`，经过两层 Transformer。问题和候选可以相互读取；头内没有候选序号embedding、位置编码或因果mask。
3. **每个候选使用同一个打分函数。** 将更新后的候选向量与更新后的问题向量拼成512维，经过 `LayerNorm → Linear(512,256) → GELU → Linear(256,1)`，每行输出一个logit。K只改变输入行数，最后一层始终输出一个标量。
4. **在候选之间归一化。** softmax产生概率。是非题使用false／true两个候选，评分题使用按顺序排列的等级。

保持ID与文本不变，仅重排候选，理想计算中的输出随之重排。增删候选会改变交互和归一化，因此已有候选的分数也可能变化。候选数量可变，评分函数和参数共用。

代码保留完整因果参考路径、前缀缓存推理和带 DeltaNet 分支状态的树形执行。正式模型主干使用BF16；导出的LoRA、集合头与交互gate张量为FP32，集合头和末层交互的关键归约也保持FP32。

[详细架构公式](docs/architecture.md) · [交互式中文图解](docs/decision-head.html#set-head)

## 评测结果

准确率%，模型版本、题目范围与推理设置固定；clean子集对应Kev README采用的口径。**加粗只比较 Qev 与 Kev，标出两者中较高的成绩；Jev 仅作参考。**

**精度对比：Kev-9B 使用 FP32，Qev-9B 的主干计算使用 BF16。** Qev 的决策头和关键归约仍使用 FP32，导出的 LoRA 张量也保存为 FP32；下表是在不同精度设置下得到的结果。

| 评测 | Qev-9B | Kev-9B | Qwen3.5-9B-Base | Jev（参考） |
|---|---:|---:|---:|---:|
| decision_dev · clean | **87.42** | 87.18 | 77.69 | 84.49 |
| transfer_dev · clean | **83.99** | 82.16 | 74.39 | 85.67 |
| MMLU-Pro · 1000题 | **54.60** | 51.10 | 50.40 | 83.50 |
| SemIf · 144道手写题 | **93.75** | 90.97 | 90.28 | 96.53 |
| scienthoon · 873题 | 72.28 | **75.49** | 68.84 | 75.26 |
| WANLI · 256题 | **72.66** | 70.31 | 67.97 | 75.78 |
| JevBench公开题 · 231题 | **81.39** | 75.76 | 75.76 | 85.71 |

<img src="assets/jevbench.svg" alt="Qev-9B（BF16主干）188/231，Kev-9B（FP32）175/231；Jev仅作参考" width="100%">

JevBench由我们运行，Jev通过官方服务调用；其余评测中Kev和Jev的数字来自固定版本的作者报告。Kev使用FP32，Qev和底座为BF16；Kev在MMLU-Pro中未作答的8题计错。这里是公开题的argmax准确率，不是JevBench官方综合榜单分数。

Qev正式检查点只有一个seed。数据、LoRA rank和训练配方同时变化，不能把整体差距单独归因于架构。当前单seed消融中，关闭末层候选交互和集合头attention也得到相近准确率；scienthoon上Qev低于Kev。[完整结果、消融与限制](docs/evaluation.md)

## 使用自己的数据训练

在上述请求格式的每个问题上增加 `label`：Choice填候选ID，Noul填布尔值，Score填从0开始的等级编号。相关或改写样本共用 `group_id`。

```bash
python -m qev.prepare \
  --input examples/train.jsonl --validation examples/dev.jsonl \
  --out data/support

python -m qev.train \
  --config configs/qev-9b-finetune.json \
  --data data/support --out runs/support \
  --init-checkpoint checkpoints/qev-9b
```

示例数据只用于说明格式，数量不足以评价效果。`--init-checkpoint`加载LoRA与决策头，重新开始优化器和调度；`--resume`恢复同一次训练及其原数据校验。省略初始化参数则从Qwen底座开始。

[正式配置](configs/qev-9b.json)使用四卡、全局batch 32、两轮训练、rank 64、seed 17和收尾数据分区，需要研究数据的对应分区。完整研究训练集未随代码打包。[训练说明](docs/training.md) · [数据构成](docs/data.md)

## 验证与贡献

```bash
python -m pytest -q
python scripts/check_release.py
```

测试覆盖候选换序、问题隔离、参考／缓存／树形路径、梯度、训练／恢复、可移植检查点与数据隔离。FSDP测试需要两张CUDA GPU，CPU运行会跳过；本次实际验证范围见[验证记录](docs/validation.md)。

Qev基于Qwen，并参考[Jared Palmer的Kev](https://github.com/jaredpalmer/kev)实现中的标记约定、文本渲染、LoRA目标和缓存分叉方式。代码按[Apache-2.0](LICENSE)发布，来源与署名见[NOTICE](NOTICE)和[provenance.json](provenance.json)。权重与数据保留各自适用的条款。[贡献指南](CONTRIBUTING.md) · [模型卡](docs/model-card.md)
