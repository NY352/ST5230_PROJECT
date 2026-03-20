# ST5230 Project - Claude Code Context

## 项目状态
Pipeline 代码已重构完成，统一走 OpenRouter。尚未运行。统计分析和可视化尚未开始。

## 架构
- `config.py`: 全局配置 + OpenRouter 客户端 + 工具函数（call_llm, JSON I/O, 断点续传）
- `src/data_loader.py`: HuggingFace 数据加载 + 采样1000条 + 格式统一化
- `src/paraphraser.py`: 用 Llama-3.3-70B 生成5种改写（lexical/syntactic/style/context/translation），支持断点续传
- `src/evaluator.py`: 向目标模型发请求，提取 accuracy + logprob，支持断点续传
- `run.py`: 统一入口（prepare / paraphrase / evaluate）

## 运行步骤
1. `pip install -r requirements.txt`
2. `python run.py prepare` — 下载并采样数据
3. `python run.py paraphrase` — 生成改写（最耗时，支持中断重跑）
4. `python run.py evaluate` — 评估模型
5. 统计分析 + 可视化（尚未规划）

## 关键信息
- 改写模型: meta-llama/llama-3.3-70b-instruct (OpenRouter)
- 评估模型: openai/gpt-4o-mini, qwen/qwen3.5-27b, moonshotai/kimi-k2 (全部 OpenRouter)
- 数据集: CommonsenseQA, ARC-Challenge, MMLU，各1000条
- .env 中只需 OPENROUTER_API_KEY
