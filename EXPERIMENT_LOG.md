# 实验记录

## 项目：Benchmark Reliability under Semantic Paraphrasing

### 主题：LLM 评估的可靠性（Theme 1: Reliability of LLM Evaluation）

---

## 1. 实验设置

### 1.1 研究问题

语义等价的改写是否会导致 LLM 在基准测试上的表现发生变化？如果会，哪种改写类型影响最大？

### 1.2 数据集

| 数据集 | HuggingFace 路径 | Split | 采样数量 | 随机种子 |
|--------|------------------|-------|----------|----------|
| CommonsenseQA | `commonsense_qa` | validation | 1000 | 42 |
| ARC-Challenge | `allenai/ai2_arc` / ARC-Challenge | test | 1000 | 42 |
| MMLU | `cais/mmlu` / all | test | 1000 | 42 |

采样方式：`random.Random(42).sample(range(len(ds)), 1000)`，采样后按原始索引排序。

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
- 参数：`max_tokens=512`，`temperature=0.7`

**评估目标模型：**

| 模型 | API ID | top_logprobs |
|------|--------|-------------|
| GPT-4o-mini | `openai/gpt-4o-mini` | 5 |
| Qwen3.5-27B | `qwen/qwen3.5-27b` | 5 |
| Kimi-K2 | `moonshotai/kimi-k2` | 5 |

评估参数：`max_tokens=1`，`temperature=0`（贪心解码，确保可复现）。

### 1.5 评估指标

- **Accuracy（准确率）**：模型预测的答案字母是否与正确答案一致
- **Ground-truth logprob（正确答案对数概率）**：模型对正确答案 token 赋予的 log-probability，从 `top_logprobs`（前 5）中提取

### 1.6 评估条件

每个模型在每个数据集上评估 6 个条件：
- `baseline`（原始问题）
- `lexical`、`syntactic`、`style`、`context`、`translation`（改写后的问题）

评估阶段总 API 调用量：3 模型 × 3 数据集 × 6 条件 × 1000 条 = 54,000 次。

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

共进行了四轮 prompt 优化。

**V1（初始版本）：** 每种类型给出基本改写指令。问题：类型间交叉污染（如 lexical 也改变了句法结构），未保护空白填充题。

**V2（严格类型分离）：** 添加明确规则隔离各类型（如 lexical 增加 "ONLY replace individual words"）。问题：仍有答案泄露和句子丢失。

**V3（添加反例）：** 增加具体反例（如 "'hamburger' stays 'hamburger', NOT 'sandwich'"），增加空白题保护（"If the original ends with '...the what?', the rewrite must also end with '...the what?'"）。问题：context 类型的 intro 句子高度重复，多句问题仍会丢句。

**V4（最终生产版本）：** 增加多句保护（"If the input has multiple sentences, you MUST keep ALL sentences"），增加 context 多样性指令（"Pick a RANDOM topic each time — vary across weather, geography, history, science, sports, food, etc."），增加 context intro 必须为陈述句的约束（"The intro sentence MUST be a declarative statement, NOT a question."）。

### 3.2 最终 Prompt

完整 prompt 内容见 `config.py` 第 74–140 行。

### 3.3 小规模试点测试（每种类型 60 条，CommonsenseQA）

使用 V4 prompt 在正式全量运行前进行试点测试：

| 类型 | 测试数 | 问题数 | 问题率 | 详情 |
|------|--------|--------|--------|------|
| lexical | 60 | 0 | 0% | — |
| syntactic | 60 | 6 | 10% | sentence_dropped（从句合并） |
| style | 60 | 2 | 3.3% | sentence_dropped |
| context | 60 | 1 | 1.7% | original_modified |
| translation | 60 | 0 | 0% | — |

---

## 4. 正式运行：CommonsenseQA

### 4.1 原始质量检测（每种类型 1000 条）

| 类型 | 总数 | 有问题条目 | 问题率 | 问题分布 |
|------|------|-----------|--------|----------|
| lexical | 1000 | 3 | 0.3% | answer_leaked: 3 |
| syntactic | 1000 | 90 | 9.0% | sentence_dropped: 87, question_form_lost: 2, answer_leaked: 1 |
| style | 1000 | 31 | 3.1% | sentence_dropped: 27, answer_leaked: 4 |
| context | 1000 | 35 | 3.5% | original_modified: 19, context_dropped: 16 |
| translation | 1000 | 5 | 0.5% | sentence_dropped: 5 |

**关于 syntactic 的 sentence_dropped：** 87 条主要是多句问题被模型合并为带从句的复合句（如 "Bob lives here. Where does he live?" → "Where does Bob, who lives here, live?"）。这属于合法的句法变换，语义完整保留，因此**不作为错误**处理，不进行过滤。

**关于 context 的 original_modified：** 19 条中大部分是模型自动修正了原文的拼写错误或格式问题（如多余空格、"Chrismas" → "Christmas"）。虽然改动很小，但违反了逐字保留的要求，需要过滤。

### 4.2 质量过滤

过滤脚本：`src/quality_filter.py`

**过滤标准（仅过滤硬伤）：**

| 检查项 | 适用类型 | 说明 |
|--------|---------|------|
| question_form_lost | 除 context 外所有类型 | 原文以 `?` 结尾但改写没有 |
| answer_leaked | 所有类型 | 正确答案文本出现在改写中但不在原文中 |
| blank_filled | lexical | `the/a what?` 模式在改写中丢失 |
| original_modified | context | 原始问题未被逐字保留 |
| not_chinese | translation | CJK 字符少于 3 个 |
| too_short | translation | 输出少于 5 个字符 |

**不过滤的情况：** syntactic/style 的句子合并（属于合法的从句重构）。

**过滤结果（CommonsenseQA）：**

| 类型 | 过滤前 | 过滤后 | 移除数 | 移除率 |
|------|--------|--------|--------|--------|
| lexical | 1000 | 997 | 3 | 0.3% |
| syntactic | 1000 | 997 | 3 | 0.3% |
| style | 1000 | 996 | 4 | 0.4% |
| context | 1000 | 963 | 37 | 3.7% |
| translation | 1000 | 1000 | 0 | 0% |

过滤后的文件保存为 `data_paraphrased/{dataset}_{type}_filtered.json`。

---

## 5. 进度追踪

| 步骤 | 状态 | 备注 |
|------|------|------|
| 数据准备（3 个数据集） | 已完成 | 每个数据集采样 1000 条，seed=42 |
| 改写：CommonsenseQA | 已完成 | 5 种类型 × 1000 条，已过滤 |
| 改写：ARC-Challenge | 未开始 | — |
| 改写：MMLU | 未开始 | — |
| 模型评估 | 未开始 | 3 模型 × 3 数据集 × 6 条件 |
| 统计分析 | 未开始 | 配对检验、置信度退化分析 |
| 可视化 | 未开始 | 小提琴图、热力图等 |
| 论文撰写 | 未开始 | — |

---

## 6. 文件结构

```
ST5230_Project/
├── config.py                  # 全局配置、API 客户端、prompt、工具函数
├── run.py                     # CLI 入口：prepare / paraphrase / filter / evaluate
├── requirements.txt           # 依赖
├── README.md                  # 项目说明
├── EXPERIMENT_LOG.md          # 实验记录（本文件）
├── src/
│   ├── data_loader.py         # HuggingFace 数据加载 + 采样
│   ├── paraphraser.py         # 改写生成（支持断点续传）
│   ├── evaluator.py           # 模型评估（accuracy + logprob，自动优先使用过滤后文件）
│   └── quality_filter.py      # 基于规则的质量过滤
├── test_paraphrase.py         # 试点测试脚本（每种类型 60 条）
├── data/
│   └── sampled/               # 采样后的数据集（每个 1000 条）
│       ├── commonsense_qa.json
│       ├── arc_challenge.json
│       └── mmlu.json
├── data_paraphrased/          # 改写输出
│   ├── commonsense_qa_{type}.json          # 原始（每种 1000 条）
│   └── commonsense_qa_{type}_filtered.json # 过滤后
└── results/                   # 评估结果（待生成）
```

---

## 7. 技术细节

- 所有 API 调用通过 OpenRouter（`https://openrouter.ai/api/v1`）
- 重试机制：指数退避，最多 5 次，基础延迟 2 秒
- 断点续传：每条结果单独追加写入 JSON 文件，重启时跳过已完成的 ID
- 评估 prompt 格式：`{question}\n\n{choices}\n\nAnswer with only the letter of the correct answer.`
- Logprob 提取：从 `top_logprobs`（前 5 个）中查找正确答案 token 的 logprob
