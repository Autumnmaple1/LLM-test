"""探针：pro 流式返回的 delta 字段结构——看思考内容是否输出、字段叫什么"""
import time
from llm_client import LLMClient
from bare_bench import load, sample, fmt_mmlupro

df, dev, letters = load("mmlupro")
df = sample(df, 500)
rows = {str(r.get("question_id")): fmt_mmlupro(r["question"], r["options"])
        for _, r in df.iterrows() if str(r.get("question_id")) in {"3542"}}

c = LLMClient("deepseek/deepseek-v4-pro")
t0 = time.time()
try:
    stream = c.client.chat.completions.create(
        model=c.model, messages=[{"role": "user", "content": rows["3542"]}],
        temperature=0.0, max_tokens=40000, stream=True, timeout=90)
    n = 0
    for chunk in stream:
        if not chunk.choices:
            continue
        n += 1
        if n <= 15:
            d = chunk.choices[0].delta
            print(f"chunk#{n} delta字段={list(d.model_fields_set) if hasattr(d,'model_fields_set') else dir(d)[:20]}", flush=True)
            print(f"  content={getattr(d,'content',None)!r}", flush=True)
            print(f"  reasoning_content={getattr(d,'reasoning_content',None)!r}", flush=True)
        if n == 15:
            print("... 继续收集中 ...", flush=True)
    print(f"总计 chunks={n} 耗时={time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"异常 {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
