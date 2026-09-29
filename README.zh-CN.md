<p align="center">
  <img src="assets/banner.svg" alt="Qev — 基于 Qwen 的决策模型" width="100%">
</p>

<p align="center">
  <a href="pyproject.toml"><img src="assets/badges/python.svg" alt="Python 3.12"></a>
  <a href="docs/model-card.md"><img src="assets/badges/model.svg" alt="基于 Qwen3.5"></a>
  <a href="LICENSE"><img src="assets/badges/license.svg" alt="Apache-2.0"></a>
</p>

<p align="center"><a href="README.md">English</a> | <strong>简体中文</strong> | <a href="https://huggingface.co/AustinFu/Qev-9B">🤗 模型权重</a></p>

**Qev 是从 Qwen 微调而来的决策模型。** 给定上下文、问题和候选答案，模型直接返回选择及各选项的概率。同一套模型支持 **Choice 选择、Noul 是非判断、Score 序数评分**。

本仓库提供模型架构、训练与评测代码、Python 接口和检查点工具。你可以在自己的数据上训练，也可以加载已有的 Qev 检查点进行推理。

| 从这里开始 | 可以做什么 |
|---|---|
| **[训练模型](#训练)** | 准备标注数据，从 Qwen 底座训练，或在 Qev 检查点上继续微调 |
| **[运行模型](#推理)** | 通过 Python 或 JSONL 接口，获取选项概率和决策 |

## 演示

开头预留贪吃蛇与 Crafter 的实际运行录屏，下面的卡片是待替换的展示位置。

<!-- DEMO SLOTS: replace each placeholder src with the real GIF/poster; optionally link it to a full recording. -->
<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/demos/snake-placeholder.svg" alt="贪吃蛇真实录屏待补充" width="100%">
      <br><strong>贪吃蛇 · 连续动作选择</strong>
    </td>
    <td width="50%" align="center">
      <img src="assets/demos/crafter-placeholder.svg" alt="Crafter真实录屏待补充" width="100%">
      <br><strong>Crafter · 生存与建造</strong>
    </td>
  </tr>
</table>

[录屏替换位置与说明](docs/demos.md)。

## 安装

使用 Python 3.12。在仓库根目录创建环境，激活后安装适合硬件的 PyTorch 2.8.0，再安装 Qev：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# 先在这个环境中安装适合硬件的 PyTorch 2.8.0。
python -m pip install -e .
```

开发和测试使用 `python -m pip install -e '.[test]'`。确认环境后，可先运行无需下载模型的 CPU 短试跑：

```bash
python scripts/smoke.py --out runs/smoke
```

它使用随机初始化的小型 Qwen 验证数据准备、训练、续训和推理流程；完整安装与设备说明见[训练文档](docs/training.md)。

## 训练

### 准备数据

训练数据沿用推理请求中的 `state` 和 `questions`，在每个问题上增加 `label`。Choice 填候选 ID，Noul 填布尔值，Score 填从 0 开始的等级编号。相关样本应共用 `group_id`。

| 数据 | 内容 | 入口 |
|---|---|---|
| 随包示例 | 6 条训练请求、2 条验证请求，覆盖三种任务 | [examples/](examples/README.md) |
| 自己的数据 | 带标签或软目标的 JSONL 请求 | [数据格式](docs/data.md) |
| 正式研究配方 | 主分区 34,546 条，收尾分区 1,783 条；完整数据尚未随代码分发 | [数据构成](docs/data.md#research-recipe-and-availability) |

```bash
python -m qev.prepare \
  --input examples/train.jsonl --validation examples/dev.jsonl \
  --out data/support
```

未提供独立验证文件时，工具按 `group_id` 做确定性划分，并检查训练／验证之间的 ID、分组和精确输入重合。随包示例用于熟悉流程，数量不足以评价训练收益。

### 监督微调

在 CUDA GPU 上，从已有的 Qev-9B 检查点开始新的领域训练：

```bash
python -m qev.train \
  --config configs/qev-9b-finetune.json \
  --data data/support --out runs/support \
  --init-checkpoint AustinFu/Qev-9B
```

`--init-checkpoint` 加载模型参数，重新建立优化器与学习率计划；`--resume` 恢复同一次训练及其原数据校验。省略初始化参数则从配置中的 Qwen 底座开始。

[正式四卡配置](configs/qev-9b.json)使用 rank 64、全局 batch 32、两轮训练和收尾分区混入。单卡、多卡、断点恢复与全参训练见[训练指南](docs/training.md)。

## 模型与检查点

| 模型 | 底座与结构 | 当前入口 |
|---|---|---|
| **Qev-9B** | Qwen3.5-9B-Base，rank-64 LoRA，256 维两层集合决策头 | [Hugging Face · 下载](https://huggingface.co/AustinFu/Qev-9B) |

Qev-9B 在通用决策数据上训练，并在训练后半程加入对齐题与规则判断题，重复学习三次。下载包包含 LoRA、决策头、候选交互参数、tokenizer 和模型配置。

约 690 MiB 的检查点会自动下载，加载器另行获取Qwen 底座。本地下载、检查点导出与底座缓存配置见[检查点指南](docs/checkpoints.md)；完整模型信息见[模型卡](docs/model-card.md)。

## 推理

### Python 接口

模型在进程内加载一次，随后可以反复提交请求：

```python
from qev import Qev

model = Qev.from_pretrained(
    "AustinFu/Qev-9B", device="cuda"
)
answers = model.predict({
    "state": "同一订单被扣款两次，请立即处理。",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "应该由哪个团队处理？",
            "criteria": {"billing": "账单和退款", "shipping": "物流配送"},
        }
    },
})
print(answers["department"]["choice"])
print(answers["department"]["probabilities"])
```

是非问题使用 `noul`，返回 `P(true)`；评分问题使用 `score`，返回等级分布及其期望。[三种任务的完整示例](examples/requests.jsonl)与[输入输出约定](docs/data.md)。

### JSONL 批量推理

```bash
python -m qev.predict \
  --checkpoint AustinFu/Qev-9B \
  --input examples/requests.jsonl --out runs/predictions.jsonl \
  --device cuda --weights-dtype checkpoint
```

默认使用共享前缀缓存；复现正式评测的执行路径时加 `--reference`。输出文件使用新名字，精度与长度选项见[加载说明](docs/checkpoints.md)。

## 模型怎样作出决策

<p align="center">
  <img src="assets/architecture.zh-CN.svg" alt="Qev将消息、问题和候选答案编码为数值摘要，再经过四步决策头得到选项分数。图内解释候选答案与e₁、e₂、e₃的含义。" width="100%">
</p>

Qev 按 **state → question → candidate** 组织输入。候选分支读取共享上下文，各自形成摘要；集合决策头将问题与候选摘要结合，为不同数量的选项使用同一个打分函数。实现同时提供完整因果参考路径、前缀缓存和带 DeltaNet 分支状态的树形执行。

[模型设计与公式](docs/architecture.md) · [交互式决策头图解](docs/decision-head.html#set-head)

## 评测

**精度对比：Qev-9B 的主干计算使用 BF16，Kev-9B 使用 FP32。** Qev 的决策头保持 FP32。

<p align="center">
  <img src="assets/evaluation.svg" alt="七组评测的Qev与Kev成组柱状图，柱顶为准确率，横轴标明题量。" width="100%">
</p>

| 评测 | Jev（参考） | Qwen3.5-9B-Base | Qev-9B | Kev-9B |
|---|---:|---:|---:|---:|
| decision_dev · clean | 84.49 | 77.69 | **87.42** | 87.18 |
| transfer_dev · clean | 85.67 | 74.39 | **83.99** | 82.16 |
| MMLU-Pro · 1000题 | 83.50 | 50.40 | **54.60** | 51.10 |
| SemIf · 144道手写题 | 96.53 | 90.28 | **93.75** | 90.97 |
| scienthoon · 873题 | 75.26 | 68.84 | 72.28 | **75.49** |
| WANLI · 256题 | 75.78 | 67.97 | **72.66** | 70.31 |
| JevBench公开题 · 231题 | 85.71 | 75.76 | **81.39** | 75.76 |

表中为准确率（%），加粗标出 Qev 与 Kev 中的较高成绩；Jev 与 Qwen 底座作为参考。

Qev-9B 在 JevBench 公开 231 题上答对 **188 题（81.39%）**，Kev-9B 为 175 题（75.76%）。这里展示的是公开题准确率，不是 JevBench 官方综合分数。

[完整结果、消融与评测设置](docs/evaluation.md) · [机器可读指标](results/benchmarks.json) · [231 题原始预测](results/qev-9b/jevbench-predictions.jsonl)

## 文档与贡献

[架构](docs/architecture.md) · [训练](docs/training.md) · [检查点](docs/checkpoints.md) · [数据与输出](docs/data.md) · [评测](docs/evaluation.md) · [贡献指南](CONTRIBUTING.md)

运行 `python -m pytest -q` 和 `python scripts/check_release.py` 检查代码与文档；实际验证范围和 GPU 跳过项见[验证记录](docs/validation.md)。

Qev 基于 Qwen，并参考 [Jared Palmer 的 Kev](https://github.com/jaredpalmer/kev) 中的标记约定、文本渲染、LoRA 目标和缓存分叉方式。Qev 的代码、微调权重、文档和原创图示采用 [Apache-2.0](LICENSE)。Qwen、Kev、JevBench 与外部依赖的署名和许可范围见[第三方声明](THIRD_PARTY_NOTICES.md)。
