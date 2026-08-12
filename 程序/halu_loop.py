"""HaluEval 检测-降低闭环：对 question+answer 打可疑度 -> 触发修正 -> 三组指标
HaluEval 每条: knowledge/question/right_answer/hallucinated_answer (幻觉率50%, 标签直接可用)
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from llm_client import get_client
from truthfulqa_common import load_cache, save_cache
from benchmark_truthfulqa_gen import llm_judge

RESULT_DIR = Path(__file__).parent / "results"
MAIN = "deepseek/deepseek-v4-pro"
DS = "deepseek/deepseek-v4-flash"

SAMP = "基于下面的知识回答用户的提问，只陈述知识中支持的事实，不要编造。\n知识：{k}\n问题：{q}\n回答："
CONSIST = "两个回答在核心事实上是否一致？只输出 CONSISTENT 或 CONTRADICT。\n回答A：{a}\n回答B：{b}\n判定："
REVISE = "下面的回答可能与知识不符。请基于知识给出一个明确的最终回答，直接陈述事实，不要使用'无法确定''可能'等模糊表述。\n知识：{k}\n问题：{q}\n原回答：{a}\n最终回答："
# CoVe（方案B）：保留原回答 -> 分解声明 -> 对照知识验证 -> 有错才修正
DECOMP = "将下面的回答分解为独立的原子事实陈述，每行一个，以'- '开头，不要输出其他内容。\n回答：{a}"
VERIFY = "判断下列陈述是否被【知识】支持。每行输出 SUPPORT / NOT_SUPPORT / NOT_CHECKABLE，格式：<陈述> → <判定>。\n知识：{k}\n陈述：\n{c}"
COVE_REVISE = "下面的回答包含与知识不符的陈述。请基于知识给出修正后的最终回答，直接陈述事实，不要使用模糊表述。\n知识：{k}\n问题：{q}\n原回答：{a}\n验证结果：{v}\n最终回答："

_clients = {}
def C(model, temp, mtok=40000):
    key = (model, temp)
    if key not in _clients:
        _clients[key] = get_client(model, temperature=temp, max_tokens=mtok)
    return _clients[key]

def gen(model, temp, prompt, mtok=40000):
    return C(model, temp, mtok).generate([{"role": "user", "content": prompt}], temperature=temp, max_tokens=mtok)

def consistent(a, b):
    r = gen(DS, 0.0, CONSIST.format(a=a, b=b), mtok=40000) or ""
    return "CONSISTENT" in r.upper()

def detect(it, answer):
    # 语义熵：待检回答 vs (pro 高温重答, flash 跨模型独立答) 的一致性 -> 不一致越多越可疑
    a1 = gen(MAIN, 1.0, SAMP.format(k=it["knowledge"], q=it["question"]))
    a2 = gen(DS, 0.7, SAMP.format(k=it["knowledge"], q=it["question"]))
    score = 1 - (consistent(answer, a1) + consistent(answer, a2)) / 2
    return {"score": score, "alt1": a1, "alt2": a2}

def reduce_a(it, answer, score, thresh):
    """方案A：跨模型重生成——可疑直接让 DS 重答，两模型一致才输出原回答，矛盾则生成修正版"""
    if score < thresh:
        return {"final": answer, "triggered": False}
    fresh = gen(DS, 0.3, SAMP.format(k=it["knowledge"], q=it["question"]))
    if consistent(answer, fresh):
        return {"final": answer, "triggered": True}
    rev = gen(MAIN, 0.0, REVISE.format(k=it["knowledge"], q=it["question"], a=answer), mtok=40000)
    return {"final": rev, "triggered": True}

def reduce_b(it, answer, score, thresh):
    """方案B：CoVe 式修正——保留原回答，分解声明对照知识验证，有错才修正"""
    if score < thresh:
        return {"final": answer, "triggered": False}
    claims_raw = gen(MAIN, 0.0, DECOMP.format(a=answer), mtok=40000) or ""
    claims = [l.strip().lstrip("-•* ").strip() for l in claims_raw.splitlines()
              if l.strip().startswith("-") and len(l.strip()) > 3]
    if not claims:
        return {"final": answer, "triggered": True}  # 分解失败，保守保留原回答
    ver = gen(DS, 0.0, VERIFY.format(k=it["knowledge"], c="\n".join(f"- {c}" for c in claims)), mtok=40000) or ""
    if "NOT_SUPPORT" not in ver.upper():
        return {"final": answer, "triggered": True}  # 全部被支持，无需修正
    rev = gen(MAIN, 0.0, COVE_REVISE.format(k=it["knowledge"], q=it["question"], a=answer, v=ver), mtok=40000)
    return {"final": rev, "triggered": True}

def aurc(scores, labels):
    order = np.argsort(scores)
    risks, covs = [], []
    err = 0
    for i, idx in enumerate(order):
        if labels[idx]:
            err += 1
        covs.append((i + 1) / len(labels))
        risks.append(err / (i + 1))
    return float(np.trapezoid(risks, covs))

def load_halu(n):
    items = [json.loads(l) for l in open(Path(__file__).parent / "data_halu" / "qa_data.json", encoding="utf-8") if l.strip()]
    return items[:n]

def run(max_samples, thresh, workers=24, reduce_mode="a"):
    raw = load_halu(max_samples)
    reducer = reduce_a if reduce_mode == "a" else reduce_b
    # 每条样本生成两个检测任务：right(0) / hallucinated(1)
    tasks = []
    for i, it in enumerate(raw):
        tasks.append({"it": it, "answer": it["right_answer"], "label": 0})
        tasks.append({"it": it, "answer": it["hallucinated_answer"], "label": 1})

    dets = {}
    fails = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(detect, t["it"], t["answer"]): t for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                dets[(id(t["it"]), t["label"])] = f.result()
            except Exception as e:
                fails.append((id(t["it"]), t["label"]))
                print(f"  [跳过] 检测失败: {type(e).__name__}")
    tasks = [t for t in tasks if (id(t["it"]), t["label"]) in dets]
    if not tasks:
        print("全部失败")
        return
    print(f"有效检测任务 {len(tasks)} (跳过 {len(fails)})")

    judge = C(DS, 0.0, mtok=40000)
    # 主回答正确率（降低前）：answer 与 right_answer 语义一致（并行）
    def one_judge(t):
        key = (id(t["it"]), t["label"])
        d = dets[key]
        ok_main = 1 if llm_judge(judge, t["it"]["question"], t["it"]["right_answer"], t["answer"]) == "TRUE" else 0
        r = reducer(t["it"], t["answer"], d["score"], thresh)
        ok_final = 1 if llm_judge(judge, t["it"]["question"], t["it"]["right_answer"], r["final"]) == "TRUE" else 0
        return key, ok_main, ok_final, r
    main_ok, final_ok, finals, trig = {}, {}, {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_judge, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                key, ok_main, ok_final, r = f.result()
                main_ok[key] = ok_main
                final_ok[key] = ok_final
                finals[key] = r
                trig += 1 if r["triggered"] else 0
            except Exception as e:
                fails.append(futs[f])
                print(f"  [跳过] 判定/修正失败: {type(e).__name__}")
    tasks = [t for t in tasks if (id(t["it"]), t["label"]) in finals]
    if not tasks:
        print("全部失败")
        return
    print(f"有效 {len(tasks)} (累计跳过 {len(fails)})")

    idxs = list(range(len(tasks)))
    scores = np.array([dets[(id(t["it"]), t["label"])]["score"] for t in tasks])
    labels_arr = np.array([t["label"] for t in tasks])
    mo = np.array([main_ok[(id(t["it"]), t["label"])] for t in tasks])
    fo = np.array([final_ok[(id(t["it"]), t["label"])] for t in tasks])

    # 降低效果只看幻觉样本 (label=1)：修正后答对率
    hl = labels_arr == 1
    fixed = int(((mo == 0) & (fo == 1))[hl].sum())
    false_n = int((mo == 0)[hl].sum())
    det_calls = 3 * len(tasks)
    red_calls = sum(2 if finals[(id(t["it"]), t["label"])]["triggered"] else 0 for t in tasks) + trig
    total = det_calls + red_calls

    print(f"任务 {len(tasks)} (幻觉 {int(labels_arr.sum())} 非幻觉 {len(tasks)-int(labels_arr.sum())}) | 触发率 {trig/len(tasks)*100:.0f}%")
    print(f"[检测] AUROC={roc_auc_score(labels_arr, scores):.3f} PR-AUC={average_precision_score(labels_arr, scores):.3f} AURC={aurc(scores, labels_arr):.3f} (AURC 越小越好)")
    print(f"[降低] 幻觉样本答对率 {mo[hl].mean()*100:.1f}% -> {fo[hl].mean()*100:.1f}% | 错误修正率 {fixed}/{false_n}={fixed/max(false_n,1)*100:.0f}% | 全样本诚实率 {mo.mean()*100:.1f}% -> {fo.mean()*100:.1f}%")
    print(f"[成本] 平均 {total/len(tasks):.1f} 次调用/任务 | 触发 {trig}")

    save_cache(RESULT_DIR / f"halu_{reduce_mode}_t{thresh}.jsonl", {
        "main": MAIN, "ds": DS, "thresh": thresh, "reduce": reduce_mode, "n": len(tasks),
        "metrics": {"auroc": roc_auc_score(labels_arr, scores),
                    "pr_auc": average_precision_score(labels_arr, scores),
                    "aurc": aurc(scores, labels_arr),
                    "hl_ok_main": float(mo[hl].mean()), "hl_ok_final": float(fo[hl].mean()),
                    "fix_rate": fixed / max(false_n, 1), "trig_rate": trig / len(tasks),
                    "calls_per_task": total / len(tasks)},
        "results": [{"i": i, "label": int(labels_arr[i]), "score": float(scores[i]),
                     "main_ok": int(mo[i]), "final_ok": int(fo[i]),
                     "main": tasks[i]["answer"],
                     "final": finals[(id(tasks[i]["it"]), tasks[i]["label"])]["final"],
                     "triggered": finals[(id(tasks[i]["it"]), tasks[i]["label"])]["triggered"]} for i in idxs],
    })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--reduce", default="a", choices=["a", "b"])
    args = ap.parse_args()
    t0 = time.time()
    run(args.max_samples, args.thresh, reduce_mode=args.reduce)
    print(f"耗时 {time.time()-t0:.0f}s")
