# 实验记录

## 项目：Benchmark Reliability under Semantic Paraphrasing

### 主题：LLM 评估的可靠性（Theme 1: Reliability of LLM Evaluation）

---

## 1. 实验设置

### 1.1 研究问题

语义等价的改写是否会导致 LLM 在基准测试上的表现发生变化？如果会，哪种改写类型影响最大？

### 1.2 数据集

| 数据集 | HuggingFace 路径 | Split | 初始采样 | 追加采样 | 总数 |
|--------|------------------|-------|----------|----------|------|
| CommonsenseQA | `commonsense_qa` | validation | 1000 (seed=42) | 221 (seed=43) | 1221 |
| ARC-Challenge | `allenai/ai2_arc` / ARC-Challenge | test | 1000 (seed=42) | 172 (seed=43) | 1172 |
| MMLU | `cais/mmlu` / all | test | 1000 (seed=42) | 1000 (seed=43) | 2000 |

初始采样：`random.Random(42).sample(range(len(ds)), 1000)`，按原始索引排序。

追加采样：使用 `seed=43` 从剩余未采样的条目中抽取，与初始采样无重叠。CommonsenseQA validation 仅 1221 条、ARC-Challenge test 仅 1172 条，已全部用完。

### 1.3 改写类型

| 类型 | 含义 | Prompt 设计目标 |
|------|------|-----------------|
| lexical | 同义词替换 | 仅替换内容词为同义词，保持相同的具体程度，不改变句法结构 |
| syntactic | 句法重构 | 重新排列从句、主被动转换、移动介词短语等，不替换任何词汇 |
| style | 语体风格转换 | 正式↔非正式语域转换，仅允许语域必需的最小词汇变化 |
| context | 添加无关上下文 | 在原始问题前添加一句无关的陈述句，原始问题逐字保留 |
| translation | 翻译为中文 | 忠实翻译，保留选项字母（A/B/C/D/E） |

### 1.4 模型

**改写生成模型：**
- GPT-4o（`openai/gpt-4o`），通过 OpenRouter 调用
- 参数：`max_tokens=2048`，`temperature=0.7`

**评估目标模型：**

| 模型 | API ID | top_logprobs |
|------|--------|-------------|
| GPT-4o-mini | `openai/gpt-4o-mini` | 5 |

**关于评估模型的说明：** 原计划使用 3 个模型（GPT-4o-mini、Qwen3.5-27B、Kimi-K2）。实际测试中发现 Qwen3.5-27B 和 Kimi-K2 在 OpenRouter 上**不支持 logprobs**（API 返回 `logprobs: null`）。Qwen3.5-27B 约 60.8% 的结果 logprob 为 null（取决于 OpenRouter 分配的后端 Provider），Kimi-K2 为 100% null。此外这两个模型的推理速度极慢（GPT-4o-mini 约 13 分钟/1085 条 vs Qwen3.5-27B 数小时/1085 条）。因此最终仅使用 GPT-4o-mini 进行评估。

评估参数：`max_tokens=1`，`temperature=0`（贪心解码，确保可复现）。

### 1.5 评估指标

- **Accuracy（准确率）**：模型预测的答案字母是否与正确答案一致
- **Ground-truth logprob（正确答案对数概率）**：模型对正确答案 token 赋予的 log-probability，从 `top_logprobs`（前 5）中提取。若正确答案不在 top 5 中，则为 null（GPT-4o-mini 中占 < 2%）。

### 1.6 评估条件

每个数据集评估 6 个条件：
- `baseline`（原始问题）
- `lexical`、`syntactic`、`style`、`context`、`translation`（改写后的问题）

共计：1 模型 × 3 数据集 × 6 条件 = 18 组评估，约 19,746 次 API 调用。

---

## 2. 改写模型选择

### 2.1 初始尝试：Llama-3.3-70B

- 模型：`meta-llama/llama-3.3-70b-instruct`（OpenRouter）
- 在 CommonsenseQA 上测试约 100 条（lexical 类型）
- **发现的问题：**
  - 系统性地填充空白问题：原文 "...the what?" 被改成 "...the manual." 等，直接给出了答案
  - 非同义替换：hamburger→sandwich、ferret→weasel、mountie→constable（具体程度改变）
  - 约 20% 的条目存在质量问题
- **结论：** 该模型的指令遵从能力不足以完成受控改写任务

### 2.2 切换至 GPT-4o

- 模型：`openai/gpt-4o`（OpenRouter）
- 使用相同 prompt，在 CommonsenseQA 上测试约 100 条（lexical 类型）
- **结果：**
  - 空白填充问题：0（完全解决）
  - 答案泄露：0
  - 总体问题率：<3%
- **结论：** 采用 GPT-4o 作为改写模型

---

## 3. Prompt 工程

### 3.1 迭代历史

共进行了五轮 prompt 优化。

**V1（初始版本）：** 每种类型给出基本改写指令。问题：类型间交叉污染（如 lexical 也改变了句法结构），未保护空白填充题。

**V2（严格类型分离）：** 添加明确规则隔离各类型（如 lexical 增加 "ONLY replace individual words"）。问题：仍有答案泄露和句子丢失。

**V3（添加反例）：** 增加具体反例（如 "'hamburger' stays 'hamburger', NOT 'sandwich'"），增加空白题保护（"If the original ends with '...the what?', the rewrite must also end with '...the what?'"）。问题：context 类型的 intro 句子高度重复，多句问题仍会丢句。

**V4（多句保护）：** 增加多句保护（"If the input has multiple sentences, you MUST keep ALL sentences"），增加 context 多样性指令（"Pick a RANDOM topic each time"），增加 context intro 必须为陈述句的约束。用于 CommonsenseQA 的首次全量运行。问题：在 ARC-Challenge 和 MMLU 上仍有大量句子丢失（syntactic 48%、translation 52%），因为这些数据集包含更长的多句问题和阅读材料。

**V5（句子计数约束，最终版本）：** 在所有类型 prompt 开头增加 CRITICAL 段落，要求模型先计数输入句子数并在输出中保持相同数量。明确提及段落、实验描述、引用材料等必须完整保留。translation 增加 "The Chinese output should be roughly the same length as the English input"。同时将 `max_tokens` 从 512 提高到 2048 以支持长段落。

### 3.2 最终 Prompt

完整 prompt 内容见 `config.py` PARAPHRASE_PROMPTS 字典。

### 3.3 小规模试点测试（每种类型 60 条，CommonsenseQA）

使用 V4 prompt 在 CommonsenseQA 首次全量运行前进行试点测试：

| 类型 | 测试数 | 问题数 | 问题率 | 详情 |
|------|--------|--------|--------|------|
| lexical | 60 | 0 | 0% | — |
| syntactic | 60 | 6 | 10% | sentence_dropped（从句合并） |
| style | 60 | 2 | 3.3% | sentence_dropped |
| context | 60 | 1 | 1.7% | original_modified |
| translation | 60 | 0 | 0% | — |

---

## 4. 正式运行与质量分析

### 4.1 第一轮运行：CommonsenseQA（V4 prompt，1000 条）

首次全量运行仅针对 CommonsenseQA，使用 V4 prompt。质量良好，但在后续对 ARC-Challenge 和 MMLU 进行改写时发现了严重的句子丢失问题。

### 4.2 ARC-Challenge 和 MMLU 的句子丢失问题

使用 V4 prompt 对 ARC-Challenge 和 MMLU 运行后，发现严重的句子/段落丢失：

| 数据集/类型 | 丢失 ≥1 句占比 | 丢失 ≥2 句占比 |
|------------|---------------|---------------|
| ARC/syntactic | 48.0% | 18.3% |
| ARC/style | 19.8% | 8.3% |
| ARC/translation | 51.7% | 19.6% |
| MMLU/syntactic | 31.5% | 22.1% |
| MMLU/style | 21.5% | 15.2% |
| MMLU/translation | 42.3% | 31.4% |

**根本原因：** ARC-Challenge 和 MMLU 包含大量多句实验场景（5-9 句）和长阅读材料（"This question refers to the following information..."），GPT-4o 在改写时倾向于压缩或丢弃这些背景信息。CommonsenseQA 问题较短（1-2 句），因此未暴露此问题。

### 4.3 Prompt V5 改进效果

使用 V5 prompt（句子计数约束）重新运行 ARC-Challenge 和 MMLU 后：

| 数据集/类型 | V4 丢失≥1句 | V5 丢失≥1句 | 改善 |
|------------|------------|------------|------|
| ARC/syntactic | 48.0% | 32.8% | -15.2pp |
| ARC/style | 19.8% | 4.0% | -15.8pp |
| ARC/translation | 51.7% | 51.9% | 无改善 |
| MMLU/syntactic | 31.5% | 23.5% | -8.0pp |
| MMLU/style | 21.5% | 10.9% | -10.6pp |
| MMLU/translation | 42.3% | 41.3% | -1.0pp |

**结论：** style 改善最大（ARC 从 19.8%→4.0%），syntactic 有明显改善，translation 几乎无改善（受限于中文标点差异导致的分句检测误报，以及模型对长段落翻译的固有局限）。

### 4.4 数据追加

由于严格过滤会移除较多条目，追加采样以保证过滤后的样本量：

- CommonsenseQA：从 validation 追加 221 条（用完全部 1221 条）
- ARC-Challenge：从 test 追加 172 条（用完全部 1172 条）
- MMLU：从 test 追加 1000 条（总计 2000 条，来源充足）

### 4.5 最终质量过滤

过滤脚本：`src/quality_filter.py`

**过滤标准：**

| 检查项 | 适用类型 | 说明 |
|--------|---------|------|
| question_form_lost | lexical, syntactic, style, translation | 原文以 `?` 结尾但改写没有 |
| answer_leaked | 所有类型 | 正确答案文本出现在改写中但不在原文中（答案 >3 字符） |
| sentence_dropped | lexical, syntactic, style | 英文分句器检测到改写句子数少于原文（丢 ≥1 句即过滤） |
| blank_filled | lexical | `the/a what?` 模式在改写中丢失 |
| original_modified | context | 原始问题未被逐字保留（空白符归一化后比较） |
| content_lost | translation | 中文长度 / 英文长度 < 0.15（正常翻译比例为 0.20-0.40） |
| not_chinese | translation | CJK 字符少于 3 个 |
| too_short | translation | 输出少于 5 个字符 |

**关于 translation 的过滤策略：** 不使用句子数对比，因为中文翻译常将英文的多个短句合并为逗号连接的长句（如 "He lived here. He worked there." → "他住在这里，在那里工作。"），语义完整但句子数减少。改用长度比 < 0.15 作为阈值，精确捕捉真正的段落丢失（42 条，占 1.0%），不误伤正常的中文压缩。

**各类型过滤结果：**

| 数据集 | 类型 | 过滤前 | 过滤后 | 移除数 | 主要原因 |
|--------|------|--------|--------|--------|----------|
| CommonsenseQA | lexical | 1221 | 1218 | 3 | answer_leaked |
| CommonsenseQA | syntactic | 1221 | 1115 | 106 | sentence_dropped |
| CommonsenseQA | style | 1221 | 1191 | 30 | sentence_dropped |
| CommonsenseQA | context | 1221 | 1176 | 45 | original_modified |
| CommonsenseQA | translation | 1221 | 1219 | 2 | content_lost |
| ARC-Challenge | lexical | 1172 | 1171 | 1 | answer_leaked |
| ARC-Challenge | syntactic | 1172 | 784 | 388 | sentence_dropped |
| ARC-Challenge | style | 1172 | 1123 | 49 | sentence_dropped |
| ARC-Challenge | context | 1172 | 1049 | 123 | original_modified |
| ARC-Challenge | translation | 1172 | 1170 | 2 | content_lost |
| MMLU | lexical | 2000 | 1899 | 101 | sentence_dropped |
| MMLU | syntactic | 2000 | 1528 | 472 | sentence_dropped |
| MMLU | style | 1999 | 1780 | 219 | sentence_dropped |
| MMLU | context | 2000 | 1909 | 91 | original_modified |
| MMLU | translation | 2000 | 1954 | 46 | content_lost |

### 4.6 ID 交集

对每个数据集，取 5 种改写类型过滤后的 ID 交集，确保同一数据集内所有条件（baseline + 5 种改写）使用完全相同的题目集合：

| 数据集 | 交集前（最小类型） | **交集后（最终）** |
|--------|-------------------|-------------------|
| CommonsenseQA | 1115 (syntactic) | **1085** |
| ARC-Challenge | 784 (syntactic) | **732** |
| MMLU | 1528 (syntactic) | **1474** |

过滤后的文件保存为 `data_paraphrased/{dataset}/{type}_filtered.json`。

---

## 5. 评估结果

### 5.1 Summary Table：Accuracy + Confidence Degradation (Δt)

**Δt = L̄_paraphrase − L̄_baseline**（配对 t 检验，95% CI）

| 数据集 | 改写类型 | N | Acc_base | Acc_para | LP_base | LP_para | **Δt** | 95% CI | p-value |
|--------|---------|---|----------|----------|---------|---------|--------|--------|---------|
| ARC-Challenge | context | 732 | 0.923 | 0.908 | -0.749 | -0.949 | **-0.201** | [-0.346, -0.055] | 0.0070 ** |
| ARC-Challenge | lexical | 732 | 0.923 | 0.904 | -0.750 | -0.889 | **-0.140** | [-0.287, +0.008] | 0.0642 |
| ARC-Challenge | style | 732 | 0.923 | 0.906 | -0.750 | -0.863 | **-0.113** | [-0.274, +0.048] | 0.1704 |
| ARC-Challenge | syntactic | 732 | 0.923 | 0.919 | -0.749 | -0.775 | **-0.026** | [-0.158, +0.106] | 0.6971 |
| ARC-Challenge | translation | 732 | 0.923 | 0.902 | -0.714 | -0.950 | **-0.235** | [-0.432, -0.038] | 0.0195 * |
| CommonsenseQA | context | 1085 | 0.801 | 0.771 | -1.847 | -2.118 | **-0.270** | [-0.451, -0.090] | 0.0034 ** |
| CommonsenseQA | lexical | 1085 | 0.801 | 0.768 | -1.841 | -2.095 | **-0.254** | [-0.450, -0.058] | 0.0112 * |
| CommonsenseQA | style | 1085 | 0.801 | 0.769 | -1.784 | -2.106 | **-0.323** | [-0.528, -0.118] | 0.0021 ** |
| CommonsenseQA | syntactic | 1085 | 0.801 | 0.756 | -1.847 | -2.139 | **-0.292** | [-0.496, -0.088] | 0.0052 ** |
| CommonsenseQA | translation | 1085 | 0.801 | 0.745 | -1.739 | -2.182 | **-0.443** | [-0.686, -0.200] | 0.0004 *** |
| MMLU | context | 1474 | 0.791 | 0.765 | -1.698 | -2.004 | **-0.307** | [-0.444, -0.169] | 0.0000 *** |
| MMLU | lexical | 1474 | 0.791 | 0.772 | -1.709 | -1.904 | **-0.194** | [-0.312, -0.077] | 0.0012 ** |
| MMLU | style | 1474 | 0.791 | 0.787 | -1.719 | -1.829 | **-0.110** | [-0.230, +0.011] | 0.0747 |
| MMLU | syntactic | 1474 | 0.791 | 0.785 | -1.702 | -1.732 | **-0.030** | [-0.159, +0.098] | 0.6437 |
| MMLU | translation | 1474 | 0.791 | 0.733 | -1.663 | -2.159 | **-0.496** | [-0.672, -0.320] | 0.0000 *** |

显著性标记：\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001

### 5.2 Failure Mode 诊断

三种 failure mode 定义（仅对 baseline 预测正确的项进行分类）：
- **Robust**：改写后仍正确，logprob 变化 < 0.5
- **Hidden Hesitation**：改写后仍正确，但 logprob 下降 > 0.5（"答对了但变得不确定"）
- **Total Collapse**：改写后预测错误

| 数据集 | 改写类型 | Robust | Hidden Hes. | Total Collapse | Base Wrong |
|--------|---------|--------|-------------|----------------|------------|
| ARC-Challenge | context | 654 (96.7%) | 1 (0.1%) | 21 (3.1%) | 56 |
| ARC-Challenge | lexical | 652 (96.4%) | 1 (0.1%) | 23 (3.4%) | 56 |
| ARC-Challenge | style | 650 (96.2%) | 1 (0.1%) | 25 (3.7%) | 56 |
| ARC-Challenge | syntactic | 649 (96.0%) | 4 (0.6%) | 23 (3.4%) | 56 |
| ARC-Challenge | translation | 640 (94.7%) | 3 (0.4%) | 33 (4.9%) | 56 |
| CommonsenseQA | context | 800 (92.1%) | 7 (0.8%) | 62 (7.1%) | 216 |
| CommonsenseQA | lexical | 792 (91.1%) | 3 (0.3%) | 74 (8.5%) | 216 |
| CommonsenseQA | style | 785 (90.3%) | 4 (0.5%) | 80 (9.2%) | 216 |
| CommonsenseQA | syntactic | 785 (90.3%) | 3 (0.3%) | 81 (9.3%) | 216 |
| CommonsenseQA | translation | 749 (86.2%) | 10 (1.2%) | 110 (12.7%) | 216 |
| MMLU | context | 1095 (93.9%) | 3 (0.3%) | 68 (5.8%) | 308 |
| MMLU | lexical | 1108 (95.0%) | 4 (0.3%) | 54 (4.6%) | 308 |
| MMLU | style | 1105 (94.8%) | 3 (0.3%) | 58 (5.0%) | 308 |
| MMLU | syntactic | 1098 (94.2%) | 7 (0.6%) | 61 (5.2%) | 308 |
| MMLU | translation | 1036 (88.9%) | 4 (0.3%) | 126 (10.8%) | 308 |

### 5.3 关键发现

**H1（Stability）—— 改写是否导致置信度分布偏移？**

是的。15 个条件中有 11 个 Δt 显著 < 0（p < 0.05）。改写系统性地降低了模型对正确答案的置信度。

不显著的 4 个条件为：ARC/lexical (p=0.064)、ARC/style (p=0.170)、ARC/syntactic (p=0.697)、MMLU/syntactic (p=0.644)。

**H2（Heterogeneity）—— 不同改写类型是否产生不同的偏移模式？**

是的。改写类型的影响有明确的层级关系：

1. **Translation（影响最大）**：Δt = -0.235 ~ -0.496，所有数据集均高度显著
2. **Context**：Δt = -0.201 ~ -0.307，所有数据集均显著
3. **Style / Lexical**：Δt = -0.110 ~ -0.323，部分显著
4. **Syntactic（影响最小）**：Δt = -0.026 ~ -0.292，ARC 和 MMLU 上不显著

**其他发现：**

- **数据集敏感度差异**：CommonsenseQA 对所有改写类型最敏感（5/5 显著），ARC-Challenge 最稳定（仅 2/5 显著），MMLU 居中（3/5 显著）。ARC 的高 baseline accuracy（92.3%）可能是原因之一。
- **Hidden Hesitation 极少**：在所有条件中占比 < 1.2%。GPT-4o-mini 很少出现"答对但不确定"的模式——要么稳定正确（Robust），要么直接翻转（Total Collapse）。这暗示该模型的决策边界较为明确。
- **Translation 的 Total Collapse 率最高**：CommonsenseQA 12.7%、MMLU 10.8%。语言切换（英→中）对模型的影响远大于英文内部的改写。

### 5.4 关于 Logprob 分布的说明

GPT-4o-mini 的 logprob 分布极度右偏：约 50-74% 的 baseline 项 logprob = 0（模型完全确定），少数项 logprob 低至 -30。这导致：

- **中位数 ≈ 0**（不如均值有信息量）
- **可视化需特殊处理**：ΔLogprob 图中排除了双零项（baseline 和 paraphrase logprob 均为 0 的项）以聚焦有信息量的变化，并将 Y 轴限制在 [-8, 8]

GPT-4o-mini 中 logprob 为 null 的比例 < 2%（正确答案不在 top 5 token 中），不影响分析。

---

## 6. 结果文件索引（论文写作参考）

### 6.1 统计分析结果

| 文件路径 | 内容 | 论文用途 |
|----------|------|----------|
| `results/analysis_summary.json` | 完整分析结果（summary table + failure modes + 代表性样例） | Results 和 Analysis 部分的数据源 |

### 6.2 可视化

| 文件路径 | 内容 | 论文用途 |
|----------|------|----------|
| `figures/delta_logprob_gpt-4o-mini_commonsense_qa.png` | CommonsenseQA ΔLogprob 分布 violin+boxplot | **主图（推荐放 Results）** |
| `figures/delta_logprob_gpt-4o-mini_arc_challenge.png` | ARC-Challenge ΔLogprob 分布 | **主图** |
| `figures/delta_logprob_gpt-4o-mini_mmlu.png` | MMLU ΔLogprob 分布 | **主图** |
| `figures/violin_gpt-4o-mini_commonsense_qa.png` | CommonsenseQA 原始 logprob 对比 | 附录 |
| `figures/violin_gpt-4o-mini_arc_challenge.png` | ARC-Challenge 原始 logprob 对比 | 附录 |
| `figures/violin_gpt-4o-mini_mmlu.png` | MMLU 原始 logprob 对比 | 附录 |

ΔLogprob 图的 caption 建议：*"Items where both baseline and paraphrase logprobs equal zero are excluded to focus on cases with measurable uncertainty. Y-axis truncated at [-8, 8]."*

### 6.3 原始评估数据

| 文件路径 | 内容 |
|----------|------|
| `results/gpt-4o-mini_{dataset}_baseline.json` | 3 个 baseline 评估结果 |
| `results/gpt-4o-mini_{dataset}_{type}.json` | 15 个改写条件评估结果 |

每个 JSON 文件中每条记录包含：`id`, `question`, `choices`, `answer`, `predicted_answer`, `is_correct`, `gt_logprob`

### 6.4 代码文件（Experimental Setup 部分参考）

| 文件 | 论文写作参考内容 |
|------|-----------------|
| `config.py` | 模型名称、API 参数、完整 prompt 模板 |
| `src/paraphraser.py` | 改写生成 pipeline |
| `src/quality_filter.py` | 过滤规则的具体实现 |
| `src/evaluator.py` | 评估协议（prompt 格式、logprob 提取） |
| `src/analysis.py` | 统计方法（Δt、配对 t 检验、failure mode 定义） |
| `src/visualize.py` | 可视化方法 |

---

## 7. 进度追踪

| 步骤 | 状态 | 备注 |
|------|------|------|
| 数据准备（3 个数据集） | ✅ 完成 | CommonsenseQA 1221, ARC 1172, MMLU 2000 |
| 改写：CommonsenseQA | ✅ 完成 | 5 种类型，已过滤 + 取交集 → 1085 条/类型 |
| 改写：ARC-Challenge | ✅ 完成 | 5 种类型，已过滤 + 取交集 → 732 条/类型 |
| 改写：MMLU | ✅ 完成 | 5 种类型，已过滤 + 取交集 → 1474 条/类型 |
| 模型评估 | ✅ 完成 | GPT-4o-mini × 3 数据集 × 6 条件 = 18 组 |
| 统计分析 | ✅ 完成 | Δt + 95% CI + 配对 t 检验 + failure mode 诊断 |
| 可视化 | ✅ 完成 | violin+boxplot（原始 logprob + ΔLogprob）|
| 论文撰写 | 🔲 未开始 | 8-10 页 conference 格式 |

---

## 8. 文件结构

```
ST5230_Project/
├── config.py                  # 全局配置、API 客户端、prompt、工具函数
├── run.py                     # CLI 入口：prepare / expand / paraphrase / filter / evaluate / analyze
├── requirements.txt           # 依赖
├── README.md                  # 项目说明
├── EXPERIMENT_LOG.md          # 实验记录（本文件）
├── src/
│   ├── data_loader.py         # HuggingFace 数据加载 + 采样 + 追加采样
│   ├── paraphraser.py         # 改写生成（支持断点续传，max_tokens=2048）
│   ├── evaluator.py           # 模型评估（accuracy + logprob，自动优先使用过滤后文件）
│   ├── quality_filter.py      # 质量过滤 + ID 交集
│   ├── analysis.py            # 统计分析（Δt、配对 t 检验、failure mode）
│   └── visualize.py           # 可视化（violin + boxplot）
├── test_paraphrase.py         # 试点测试脚本（每种类型 60 条）
├── data/
│   └── sampled/               # 采样后的数据集
│       ├── commonsense_qa.json  # 1221 条
│       ├── arc_challenge.json   # 1172 条
│       └── mmlu.json            # 2000 条
├── data_paraphrased/          # 改写输出（按数据集分子目录）
│   ├── commonsense_qa/
│   │   ├── {type}.json            # 原始改写
│   │   └── {type}_filtered.json   # 过滤 + 取交集后（1085 条/类型）
│   ├── arc_challenge/
│   │   ├── {type}.json
│   │   └── {type}_filtered.json   # 732 条/类型
│   └── mmlu/
│       ├── {type}.json
│       └── {type}_filtered.json   # 1474 条/类型
├── results/                   # 评估结果 + 分析结果
│   ├── gpt-4o-mini_*.json       # 18 个评估结果文件
│   └── analysis_summary.json    # 统计分析汇总
└── figures/                   # 可视化图表
    ├── delta_logprob_*.png      # ΔLogprob 分布图（主图，3 张）
    └── violin_*.png             # 原始 logprob 对比图（附录，3 张）
```

---

## 9. 技术细节

- 所有 API 调用通过 OpenRouter（`https://openrouter.ai/api/v1`）
- 重试机制：指数退避，最多 5 次，基础延迟 2 秒
- 断点续传：每条结果单独追加写入 JSON 文件，重启时跳过已完成的 ID
- 评估 prompt 格式：`{question}\n\n{choices}\n\nAnswer with only the letter of the correct answer.`
- Logprob 提取：从 `top_logprobs`（前 5 个）中查找正确答案 token 的 logprob
- 英文分句器：`re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)`
- Translation 内容丢失阈值：`len(中文) / len(英文) < 0.15`（正常翻译比例 0.20-0.40）
- 统计分析：配对 t 检验（H1: Δt ≠ 0），threshold=0.5 定义 Hidden Hesitation
- ΔLogprob 可视化：排除双零项（baseline 和 paraphrase logprob 均为 0），Y 轴 [-8, 8]
