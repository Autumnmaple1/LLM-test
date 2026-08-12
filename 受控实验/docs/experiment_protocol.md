# 受控实验协议（experiment_protocol）

> 项目：LLM 幻觉检测与缓解受控实验（2026-08-12）
> 目标：真实作答场景下，研究多次回答、回答冲突、跨模型协作能否以可控成本提升正确率，并分析冲突对错误风险的提示作用。

## 1. 研究问题
1. 单次回答正确率不高时，多次独立作答能否提升最终正确率？
2. 多次回答的冲突程度能否识别原始回答更可能出错的样本？
3. 固定成本预算下，只修正高风险回答是否优于随机修正？
4. 同模型多次采样 vs 不同模型协作，后者是否有额外收益？
5. 正确率提升是否值得其 token、费用和延迟成本？

**口径**：回答错误 ≠ 幻觉。"正确率"为主指标；仅在带可靠事实标签的数据集（HLE 有参考答案、TruthfulQA 有标准答案）讨论事实一致性。**本实验无外部证据注入**（所有 prompt 不带 knowledge/检索证据），对应"无证据开放域"设定。

## 2. 数据与抽样（固定种子 42）
| 数据集 | 抽样 | 题数 | 说明 |
|---|---|---|---|
| MMLU-Pro | 按 category 分层（14 类 ×7 + 随机补 2） | 100 | 0-shot 选择题，字母判分 |
| HLE | 纯文本（去 342 含图题）按 answer_type 分层 | 50（mc 12 + em 38） | 高难题，mc 字母 / em 归一化+LLM judge |
| TruthfulQA | MC1 随机 | 100 | 稳定错误常识边界测试 |

抽样文件：`data/{dataset}_{n}.jsonl`（含 qid/题干/选项/答案/种子元信息）。

## 3. 模型与统一设置
- 主模型 A = `deepseek/deepseek-v4-flash`（便宜、部署主力）；交叉验证模型 B = `deepseek/deepseek-v4-pro`（仅交叉验证/重答时用）
- judge = flash（thinking=False，~2 token/次）
- 温度：a0 生成 0.0；SC 采样 0.7；own 候选 a1=1.0、a2=0.0（不同提示角度暂同 prompt）、b1=0.7
- max_tokens=40000；超时 300s；空响应重试上限 50 次（1s 间隔）；并发 12-16
- 每次调用记录：prompt/completion/reasoning/total token、费用（官方定价）、延迟、重试、原始文本
- 判分：MMLU-Pro/TruthfulQA/HLE-mc 字母匹配；HLE-em 归一化匹配 + LLM judge 兜底
- 成本定价（元/M，2026-08 官网）：flash 输入1/输出2；pro 输入3/输出6

## 4. 实验组
| 组 | 方法 | 说明 |
|---|---|---|
| bare | A 单次回答 | 基线 |
| sc | A 独立答 K=3 | 选择题多数投票；开放题 judge 选最可信 |
| cove | 分解→自检→改写 | 无外部证据，自检不可靠——仅对照"自我核查是否提升/误改" |
| own | a0 + {A 高温, A 同角度, B 交叉} + 冲突风险 + 选择性修正 | 风险=与 a0 不一致候选比例；修正=多数票/judge 选（所有题真实执行） |
| low/high/cross | 消融：候选来源 = 同模型低温 / 同模型高温 / A+B | 固定候选数、提示、阈值、修正流程 |
| rarr | 未实现 | 无合规检索能力，不伪造结果 |

## 5. 对照（从 own 缓存构造，修正确实全部真实执行）
裸答（不触发） / 全量修正（全部触发） / 随机触发（×30 模拟均值±sd）/ 选择性触发（按 risk 前 k%）/ Oracle（按判分真实错误触发）。
调用口径：裸答 1 次/题；触发另 +3 次（a1+a2+b1 候选生成）。

## 6. 指标
- 正确性：原始/最终正确率、净提升、错误修正率、误改率、四格转移
- 检测：risk 分数的 AUROC/PR-AUC/AURC（label=原始回答错误）、固定触发率 Precision/Recall/F1、Top-k 错误率、risk 分布
- 成本：调用/题、token（含 reasoning 单列）、费用、延迟 P50/P95、重试率
- 统计：Wilson CI；按题 bootstrap 1000 次；选择性 vs 随机配对 bootstrap（p 值）；随机触发 ×30

## 7. 局限
- 测试集规模小（100/50/100 题），无独立验证集 → 阈值为探索性分析，不声称泛化
- Judge 偏差：HLE-em 判分用 flash 模型，可能误判
- 无证据环境：本实验不注入知识，与 RAG 场景的"证据忠实性"设定不同
- 模型版本/高温采样随机性：关键数字带 CI，未做 3 次重复（成本限制，探索性）
- CoVe 无证据自检不可靠，仅作对照
- 费用按官方定价估算（实际 API 计费可能含缓存命中折扣等差异）

## 8. 复现命令
```bash
cd controlled_exp
python3 src/sample.py            # 抽样（种子 42）
python3 src/run.py --dataset mmlupro --groups all --workers 16
python3 src/run.py --dataset truthfulqa --groups all --workers 16
python3 src/run.py --dataset hle --groups all --workers 16
python3 src/stats.py --dataset all   # 生成 results/summary/*_summary.md
```
