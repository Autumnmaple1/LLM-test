"""实验组实现（受控实验）：Bare / Self-Consistency / CoVe / 本人方案(own) / 跨模型消融
主模型 A=deepseek-v4-pro，交叉 B=deepseek-v4-flash。每题返回完整记录（calls/a0/候选/risk/final/判分）。
own 组对所有题真实执行修正流程（全量修正缓存），策略构造（随机/选择性/Oracle）从缓存做——修正均真实运行。
"""
from controlled_exp.src.client import ExpClient
from controlled_exp.src.judge import judge, letter_of, judge_equiv

A = "deepseek/deepseek-v4-flash"  # 主模型（便宜、部署主力）
B = "deepseek/deepseek-v4-pro"    # 交叉验证模型（仅在验证/重答时用）

def fmt_mc(item):
    if "options" in item:  # MMLU-Pro
        body = "\n".join(f"{'ABCDEFGHIJ'[i]}. {o}" for i, o in enumerate(item["options"]))
        return f"Answer the following multiple-choice question. Output only the letter of the correct answer.\nQuestion: {item['question']}\nOptions:\n{body}\nAnswer:"
    if "choices" in item:  # TruthfulQA MC1
        body = "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(item["choices"]))
        return f"Answer the following multiple-choice question. Output only the letter of the correct answer.\nQuestion: {item['question']}\nOptions:\n{body}\nAnswer:"
    return f"Answer the following question.\nQuestion: {item['question']}\nAnswer:"  # HLE mc question 自带选项

def fmt_open(item):
    return f"Answer the following question concisely and accurately.\nQuestion: {item['question']}\nAnswer:"

def is_mc(item):
    return item.get("answer_type", "mc") != "exactMatch"

def risk_of(dataset, item, a0, cands, judge_c):
    """冲突风险分数：与 a0 不一致的候选比例。选择题比字母，开放题 LLM judge"""
    if is_mc(item):
        l0 = letter_of(a0)
        return sum(1 for c in cands if letter_of(c) != l0) / len(cands)
    ds = []
    for c in cands:
        r = judge_c.gen(f"Are the following two answers factually equivalent? Answer YES or NO only.\nAnswer A: {a0}\nAnswer B: {c}\nEquivalent:")
        ds.append("YES" not in r["resp"].upper())
    return sum(ds) / len(ds)

def majority(items):
    from collections import Counter
    cnt = Counter(x for x in items if x)
    return cnt.most_common(1)[0][0] if cnt else None

def run_bare(item, cA, cB=None, judge_c=None):
    calls = []
    r = cA.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item))
    calls.append(r)
    correct, pred = judge(item.get("_ds"), item, r["resp"], cA)
    return {"calls": calls, "a0": r["resp"], "a0_correct": correct, "final": r["resp"], "final_correct": correct}

def run_sc(item, cA, cB=None, judge_c=None, K=3):
    calls = []
    ans = []
    for _ in range(K):
        r = cA.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item), temp=0.7)
        calls.append(r)
        ans.append(r["resp"])
    if is_mc(item):
        pred = majority([letter_of(a) for a in ans])
        final = pred if pred else ans[0]
    else:
        sel = judge_c.gen(f"Which of the following answers is the most accurate and trustworthy? Output only the number 1/2/3.\n1: {ans[0]}\n2: {ans[1]}\n3: {ans[2]}\nChoice:")
        calls.append(sel)
        idx = 0
        for ch in sel["resp"]:
            if ch.isdigit() and 1 <= int(ch) <= 3:
                idx = int(ch) - 1
                break
        final = ans[idx]
    correct, _ = judge(item.get("_ds"), item, final, judge_c)
    return {"calls": calls, "a0": ans[0], "a0_correct": bool(judge(item.get("_ds"), item, ans[0], judge_c)[0]),
            "final": final, "final_correct": correct}

DECOMP = "Decompose the following answer into independent atomic factual statements, one per line starting with '-'. Output nothing else.\nAnswer: {a}"
CHECK = "For each statement below, judge whether it is true. Output one line per statement: SUPPORTED or NOT_SUPPORTED.\nStatements:\n{s}"
REWRITE = "Some statements above may be incorrect. Give a corrected final answer to the question, stating only facts you are confident about. Avoid vague wording.\nQuestion: {q}\nOriginal answer: {a}\nCorrected answer:"

def run_cove(item, cA, cB=None, judge_c=None):
    calls = []
    r = cA.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item))
    calls.append(r)
    a0 = r["resp"]
    d = cA.gen(DECOMP.format(a=a0))
    calls.append(d)
    stmts = [l.strip().lstrip("-•* ") for l in d["resp"].splitlines() if l.strip().startswith("-") and len(l.strip()) > 3]
    if not stmts:
        final, fc = a0, bool(judge(item.get("_ds"), item, a0, cA)[0])
        return {"calls": calls, "a0": a0, "a0_correct": fc, "final": a0, "final_correct": fc}
    v = cA.gen(CHECK.format(s="\n".join(f"- {s}" for s in stmts)))
    calls.append(v)
    if "NOT_SUPPORTED" not in v["resp"].upper():
        fc = bool(judge(item.get("_ds"), item, a0, cA)[0])
        return {"calls": calls, "a0": a0, "a0_correct": fc, "final": a0, "final_correct": fc}
    rw = cA.gen(REWRITE.format(q=item["question"], a=a0))
    calls.append(rw)
    fc = bool(judge(item.get("_ds"), item, rw["resp"], cA)[0])
    return {"calls": calls, "a0": a0, "a0_correct": bool(judge(item.get("_ds"), item, a0, cA)[0]),
            "final": rw["resp"], "final_correct": fc}

def run_own(item, cA, cB, judge_c):
    """本人方案：a0 + 多角度重答(高温/不同角度/交叉模型) → 冲突风险 → 真实执行修正(全量)"""
    calls = []
    r0 = cA.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item))
    calls.append(r0)
    a0 = r0["resp"]
    cands = []
    for temp, prompt, cli in [(1.0, None, cA), (0.0, None, cA), (0.7, None, cB)]:
        r = cli.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item), temp=temp)
        calls.append(r)
        cands.append(r["resp"])
    risk = risk_of(item.get("_ds"), item, a0, cands, judge_c)
    # 修正流程（真实执行，全部题）：选择题多数票；开放题 judge 选最可信
    if is_mc(item):
        final = majority([letter_of(a0)] + [letter_of(c) for c in cands]) or a0
    else:
        sel = judge_c.gen(f"Which of the following answers is the most accurate? Output only 0/1/2/3.\n0: {a0}\n1: {cands[0]}\n2: {cands[1]}\n3: {cands[2]}\nChoice:")
        calls.append(sel)
        idx = 0
        for ch in sel["resp"]:
            if ch.isdigit() and 0 <= int(ch) <= 3:
                idx = int(ch)
                break
        final = ([a0] + cands)[idx]
    a0c = bool(judge(item.get("_ds"), item, a0, judge_c)[0])
    fc = bool(judge(item.get("_ds"), item, final, judge_c)[0])
    return {"calls": calls, "a0": a0, "a0_correct": a0c, "cands": cands, "risk": risk,
            "final": final, "final_correct": fc}

def run_ablation(item, cA, cB, judge_c, variant):
    """跨模型消融：候选来源变体，a0 固定 A temp0，只变候选生成源"""
    calls = []
    r0 = cA.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item))
    calls.append(r0)
    a0 = r0["resp"]
    if variant == "low":      src = [(cA, 0.3), (cA, 0.3)]
    elif variant == "high":   src = [(cA, 1.0), (cA, 1.0)]
    elif variant == "cross":  src = [(cA, 1.0), (cB, 0.7)]
    else:                     raise ValueError(variant)
    cands = []
    for cli, temp in src:
        r = cli.gen(fmt_open(item) if item.get("answer_type") == "exactMatch" else fmt_mc(item), temp=temp)
        calls.append(r)
        cands.append(r["resp"])
    risk = risk_of(item.get("_ds"), item, a0, cands, judge_c)
    a0c = bool(judge(item.get("_ds"), item, a0, judge_c)[0])
    return {"calls": calls, "a0": a0, "a0_correct": a0c, "cands": cands, "risk": risk}
