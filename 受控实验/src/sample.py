"""受控实验抽样：固定种子 42，可复现
- MMLU-Pro: 100 题，按 category 分层，0-shot 选择题
- HLE: 50 题纯文本，按 answer_type 分层（mc 12 + em 38）
- TruthfulQA: 100 题 MC1 随机抽样（稳定错误常识边界）
输出 data/{mmlupro,hle,truthfulqa}_{n}.jsonl，每题含题号/题干/选项/答案/抽样信息
"""
import json, random
from pathlib import Path
import pandas as pd

SEED = 42
BASE = Path(__file__).parent.parent  # controlled_exp/
ROOT = BASE.parent  # llm_benchmark/
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)
rng = random.Random(SEED)

def dump(fname, rows, meta):
    obj = {"seed": SEED, "n": len(rows), "meta": meta, "items": rows}
    (OUT / fname).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{fname}: {len(rows)} 题 -> {OUT / fname}")

# --- MMLU-Pro: 100 题，按 category 分层 ---
mp = pd.read_parquet(ROOT / "data_mmlu" / "mmlupro" / "data" / "test-00000-of-00001.parquet")
mp = mp.sample(frac=1, random_state=SEED).reset_index(drop=True)
per = 100 // mp["category"].nunique()  # 7
idx = mp.groupby("category", group_keys=False).head(per).index.tolist()
rest = [i for i in mp.index if i not in idx]
idx += rng.sample(rest, 100 - len(idx))
mp_sel = mp.loc[idx]
dump("mmlupro_100.jsonl", [{"qid": str(r["question_id"]), "question": r["question"],
                            "options": list(r["options"]), "answer": str(r["answer"]).upper(),
                            "category": r["category"]} for _, r in mp_sel.iterrows()],
     {"dataset": "MMLU-Pro", "shot": 0, "layering": "by category", "judge": "letter match"})

# --- HLE: 50 题纯文本，按 answer_type 分层 ---
hle = pd.read_parquet(ROOT / "data_hle" / "hle_test.parquet")
hle = hle[hle["image"].astype(str).str.len() == 0].sample(frac=1, random_state=SEED).reset_index(drop=True)
mc = hle[hle["answer_type"] == "multipleChoice"]
em = hle[hle["answer_type"] == "exactMatch"]
n_mc, n_em = 12, 38
mc_sel = mc.head(n_mc); em_sel = em.head(n_em)
hle_sel = pd.concat([mc_sel, em_sel])
dump("hle_50.jsonl", [{"qid": r["id"], "question": r["question"], "answer": r["answer"],
                       "answer_type": r["answer_type"], "category": r["category"],
                       "raw_subject": r["raw_subject"]} for _, r in hle_sel.iterrows()],
     {"dataset": "HLE", "text_only": True, "layering": "by answer_type (mc 12 + em 38)",
      "judge": "mc letter / em normalized + LLM judge"})

# --- TruthfulQA: 100 题 MC1 ---
tqa = [json.loads(l) for l in open(ROOT / "data_truthfulqa_mc.jsonl", encoding="utf-8") if l.strip()]
tqa_sel = rng.sample(tqa, 100)
dump("truthfulqa_100.jsonl", [{"qid": str(r["idx"]), "question": r["question"],
                               "choices": r["mc1_choices"], "labels": r["mc1_labels"],
                               "best": [c for c, l in zip(r["mc1_choices"], r["mc1_labels"]) if l][0]}
                              for r in tqa_sel],
     {"dataset": "TruthfulQA", "mode": "MC1", "note": "稳定错误常识边界测试",
      "judge": "letter match (best answer = label 1)"})
print("抽样完成")
