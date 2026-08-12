# LLM 幻觉检测与缓解 · 受控实验

真实作答场景下的受控实验：多次回答、回答冲突与跨模型协作能否以可控成本提升正确率，并提示幻觉/错误风险。

## 快速开始
```bash
python3 src/sample.py                                   # 固定种子 42 抽样
python3 src/run.py --dataset mmlupro --groups all       # 跑一个数据集全部实验组
python3 src/stats.py --dataset all                      # 生成汇总
```

## 目录
- `data/` — 固定抽样题目（含题号/答案/种子）
- `src/` — client（统一调用+成本/延迟记录）、judge（判分）、groups（实验组）、run（runner）、stats（统计）、sample（抽样）
- `results/raw/` — 逐题逐调用原始日志（断点续跑，不覆盖）
- `results/summary/` — 各数据集汇总表
- `docs/experiment_protocol.md` — 完整实验协议（设置/方法/局限）

## 实验组
Bare（单次） / Self-Consistency K=3 / CoVe（自检对照） / 本人方案（多角度重答+冲突风险+选择性修正）/ 跨模型消融（低温/高温/A+B）；RARR 未实现（无检索）。

## 关键口径
- 主模型 deepseek-v4-flash，交叉验证 deepseek-v4-pro（仅交叉验证时用），judge 用 flash（thinking=False）
- 无外部证据注入（开放域设定）；"正确率"为主指标，错误≠幻觉
- 对照（裸答/全量/随机×30/选择性/Oracle）从 own 缓存构造——所有题均真实执行过修正
- 统计：按题 bootstrap 1000 次、Wilson CI、配对检验

## 主要结论
（实验完成后由 stats.py 输出，见 results/summary/）
