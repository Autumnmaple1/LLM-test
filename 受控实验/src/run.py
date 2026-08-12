"""受控实验 runner：逐数据集逐组跑，断点续跑，进度日志
用法: python3 src/run.py --dataset mmlupro|hle|truthfulqa --groups all|bare,sc,cove,own,ablation[,low,high,cross]
结果: results/raw/{dataset}_{group}.jsonl（逐题完整记录，不覆盖）
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait as cf_wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from controlled_exp.src.client import ExpClient
from controlled_exp.src import groups

BASE = Path(__file__).parent.parent
RAW = BASE / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATA = {f.split("_")[0]: f for f in ["mmlupro_100", "hle_50", "truthfulqa_100"]}
GROUPS = {"bare": groups.run_bare, "sc": groups.run_sc, "cove": groups.run_cove,
          "own": groups.run_own, "low": lambda i, cA, cB, jc: groups.run_ablation(i, cA, cB, jc, "low"),
          "high": lambda i, cA, cB, jc: groups.run_ablation(i, cA, cB, jc, "high"),
          "cross": lambda i, cA, cB, jc: groups.run_ablation(i, cA, cB, jc, "cross")}

def load_items(dataset):
    d = json.load(open(BASE / "data" / (DATA[dataset] + ".jsonl"), encoding="utf-8"))
    items = d["items"]
    for it in items:
        it["_ds"] = dataset
    return items

def run_one(fn, item, cA, cB, judge_c):
    t0 = time.time()
    r = fn(item, cA, cB, judge_c)
    r.update({"qid": item["qid"], "dataset": item["_ds"], "wall": time.time() - t0})
    return r

def run_group(dataset, group, items, workers=12):
    cA = ExpClient(groups.A, temp=0.0)
    cB = ExpClient(groups.B, temp=0.7)
    judge_c = ExpClient("deepseek/deepseek-v4-flash", temp=0.0, thinking=False)  # 判分固定用 flash，便宜
    cache_p = RAW / f"{dataset}_{group}.jsonl"
    done = set()
    if cache_p.exists():
        for l in cache_p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["qid"])
                except Exception:
                    pass
    todo = [it for it in items if it["qid"] not in done]
    print(f"[{dataset}/{group}] 总 {len(items)}，已完成 {len(done)}，待跑 {len(todo)}", flush=True)
    if not todo:
        return
    n_ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = {ex.submit(run_one, GROUPS[group], it, cA, cB, judge_c): it for it in todo}
        starts = {f: time.time() for f in pending}
        while pending:
            done, _ = cf_wait(pending, return_when=FIRST_COMPLETED, timeout=30)
            now = time.time()
            for f in done:
                it = pending.pop(f)
                try:
                    r = f.result()
                    with open(cache_p, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_ok += 1
                except Exception as e:
                    print(f"  [{dataset}/{group}] qid={it['qid']} 失败: {type(e).__name__}: {e}", flush=True)
                if n_ok % 10 == 0:
                    print(f"  [{dataset}/{group}] 进度 {n_ok}/{len(todo)}", flush=True)
            # 单题超时跳过（pro 无限思考病理题，按未答=错处理）
            for f in [f for f in pending if now - starts[f] > 600]:
                it = pending.pop(f)
                f.cancel()
                print(f"  [{dataset}/{group}] qid={it['qid']} 超时>600s 跳过（统计按错误计）", flush=True)
    print(f"[{dataset}/{group}] 完成 {n_ok}/{len(todo)}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATA))
    ap.add_argument("--groups", default="all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-n", type=int, default=0, help="冒烟测试：只跑前 N 题")
    ap.add_argument("--skip-qids", default="", help="逗号分隔：跳过这些 qid（卡死题按未答=错）")
    args = ap.parse_args()
    items = load_items(args.dataset)
    if args.max_n > 0:
        items = items[:args.max_n]
    skip = set(q for q in args.skip_qids.split(",") if q)
    if skip:
        items = [it for it in items if it["qid"] not in skip]
        print(f"跳过 {len(skip)} 题: {sorted(skip)}")
    groups_sel = list(GROUPS) if args.groups == "all" else args.groups.split(",")
    t0 = time.time()
    for g in groups_sel:
        run_group(args.dataset, g, items, workers=args.workers)
    print(f"[{args.dataset}] 全部完成，总耗时 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
