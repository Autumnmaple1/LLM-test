# LLM 幻觉检测与降低（检测→降低闭环）

## 目录结构

```
LLM幻觉实验/
├── README.md               ← 本文件
├── 程序/                    ← 全部 Python 源码
│   ├── llm_client.py       ← 多 provider 客户端（ikun ChatGPT + DeepSeek）
│   ├── closed_loop.py      ← 检测→降低→评测闭环（TruthfulQA 版）
│   ├── halu_loop.py        ← 闭环（HaluEval 版，--reduce a|b 切降低方案）
│   ├── benchmark_truthfulqa_mc.py  ← TruthfulQA 选择题基线（MC1 / Self-Consistency）
│   ├── benchmark_truthfulqa_gen.py ← TruthfulQA 生成式基线（裸答 / CoVe）
│   ├── factscore.py        ← FActScore：原子事实分解 + 逐条验证
│   ├── halu_factscore.py   ← HaluEval FActScore 对比（并行）
│   ├── bare_bench.py       ← 裸基准：MMLU / MMLU-Pro 选择题准确率
│   ├── hle_bench.py        ← HLE 超难基准评测（断点续跑）
│   ├── hle_judge.py        ← HLE 判分（mc 字母匹配 / exactMatch + LLM judge）
│   ├── estimate_mmlu_tokens.py ← MMLU/MMLU-Pro token 估算（重建 prompt + 小样本实测校准外推 reasoning）
│   ├── truthfulqa_common.py← 数据加载 / prompt 模板 / 缓存读写
│   ├── factscore_compare.py、benchmark_mmlu.py、test.py  ← 历史脚本
│   ├── probe_reasoning.py / diagnose_mmlupro.py / diagnose2.py / grab_reasoning.py
│   │                        ← 推理模型诊断工具（流式看思考内容）
│   └── docs/               ← 文档
├── 数据/                    ← 测试集（原始数据）
│   ├── data_truthfulqa_mc.jsonl / data_truthfulqa_gen.jsonl  ← TruthfulQA 817 题
│   ├── data_halu/          ← HaluEval 10000 条（主测试集，幻觉标签天然 50%）
│   ├── data_mmlu/          ← MMLU 14042 题 + MMLU-Pro 12032 题（parquet）
│   ├── data_hle/           ← HLE 2500 题（274MB parquet）
│   └── mmlu_zero_shot.csv  ← 历史测试记录
└── 结果/                    ← 全部实验结果（已按类别整理）
    ├── 00_结果汇总.md       ← ★ 所有关键数字速查（写报告直接抄）
    ├── 01_闭环实验/         ← 检测→降低闭环结果（halu_a/b_t0.5、closed_loop_t0.67）
    ├── 02_基线实验/         ← MC1 单次 / Self-Consistency / 生成式裸答 / CoVe
    ├── 03_裸基准/           ← MMLU / MMLU-Pro / HLE 缓存与判分结果
    ├── 04_日志/             ← 全部运行日志（含 halu_ab_a/b 方案对比日志）
    └── 05_成本/             ← token_usage.jsonl（每次 API 调用的 usage 明细）
```

## 各程序用法

```bash
cd 程序/

# 1. 裸基准（模型能力参照）
python3 bare_bench.py --bench mmlu    --max-n 500 --shots 5 --model deepseek/deepseek-v4-flash
python3 bare_bench.py --bench mmlupro --max-n 500 --shots 0 --model deepseek/deepseek-v4-flash
python3 hle_bench.py  --max-n 300 --model deepseek/deepseek-v4-flash   # HLE（分层抽样、跳过含图题）
python3 hle_judge.py  --cache hle_deepseek-v4-flash_300.jsonl --judge ...  # HLE 判分

# 2. TruthfulQA 基线
python3 benchmark_truthfulqa_mc.py  --mode both    # MC1 单次 + Self-Consistency
python3 benchmark_truthfulqa_gen.py --mode both    # 生成式裸答 + CoVe

# 3. 检测→降低闭环（HaluEval 主测试集）
python3 halu_loop.py --reduce a --thresh 0.5 --n 200   # 方案 A：跨模型重生成
python3 halu_loop.py --reduce b --thresh 0.5 --n 200   # 方案 B：CoVe 式修正
python3 halu_factscore.py          # FActScore 对比（knowledge 当参考）
python3 closed_loop.py             # TruthfulQA 版闭环（负结果复现）
```

## 关键实验结果（2026-08，详见 结果/00_结果汇总.md）

| 项 | 结果 |
|---|---|
| 检测（HaluEval） | AUROC **0.873** / PR-AUC 0.863 / AURC 0.260 ✅ |
| 降低（A vs B，HaluEval 200 任务） | A 错误修正率 **80%** vs B 65%；诚实率 89.5% vs 84% |
| FActScore | 30.5% → **82.9%**（32/40 条提升） |
| 负结果（适用边界） | TruthfulQA AUROC 0.573≈随机——只捕捉"编造型"幻觉，不捕捉"稳定错误型" |
| 裸基准 | MMLU 92.5% / MMLU-Pro 84.5% / HLE 34.8%（flash） |

## 模型与配置

- **模型标识**：`provider/model` 形式——`deepseek/deepseek-v4-pro`（主模型）、`deepseek/deepseek-v4-flash`（交叉模型）、`ikun/gpt-5.6-sol`（ChatGPT 系，可选主模型）
- **密钥位置**：DeepSeek key 在 `~/.hermes/.env` 的 `DEEPSEEK_API_KEY`；ikun 代理在 `~/.hermes/config.yaml` 的 custom_providers 段（`https://api.ikuncode.cc/v1`）
- **注意**：`.env` 未包含在本打包内，运行前需自行配置；llm_client.py 顶部会自动清理坏代理环境变量
- **思考模式开关**：judge/验证等简单任务用 `thinking=False`（便宜百倍）；难题生成用 `effort=low` 抑制超长思考
