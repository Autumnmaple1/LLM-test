"""统一实验客户端：记录 token/延迟/重试/空回答/费用（每题级可复现日志）"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_client import get_client

PRICE = {"deepseek/deepseek-v4-pro": (3.0, 6.0), "deepseek/deepseek-v4-flash": (1.0, 2.0)}  # 元/M 输入/输出

class ExpClient:
    def __init__(self, model, temp=0.0, thinking=None, effort=None):
        self.model = model
        self.temp = temp
        self.thinking = thinking
        self.effort = effort
        self.cli = get_client(model, temperature=temp, max_tokens=40000)

    def gen(self, prompt, temp=None, thinking=None, effort=None):
        """返回 dict: {prompt, resp, usage, retries, latency, cost, model, temp}"""
        t0_ = __import__("time").time()
        content, usage, retries, latency = self.cli.generate(
            [{"role": "user", "content": prompt}],
            temperature=temp if temp is not None else self.temp,
            thinking=thinking if thinking is not None else self.thinking,
            effort=effort if effort is not None else self.effort,
            meta=True)
        pin, pout = PRICE.get(self.model, (1.0, 2.0))
        cost = (usage.get("prompt", 0) * pin + usage.get("completion", 0) * pout) / 1e6
        return {"prompt": prompt, "resp": content, "usage": usage, "retries": retries,
                "latency": latency, "cost": cost, "model": self.model, "temp": temp if temp is not None else self.temp}

def cost_of(calls):
    return sum(c.get("cost", 0) for c in calls)

def tokens_of(calls):
    return sum(c["usage"].get("total", 0) for c in calls if c.get("usage"))
