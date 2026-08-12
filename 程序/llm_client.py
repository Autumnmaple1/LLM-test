"""多模型 LLM 客户端
- 主模型：ikun 代理 ChatGPT（gpt-5.6-sol）
- 跨模型：DeepSeek 直连（deepseek-v4-pro）
支持 temperature / N 次采样（幻觉检测与自一致性所需）。
"""
import os
# 清掉 bashrc 里的坏代理 (http://:7897 缺主机名)，否则 httpx 走代理 DNS 全挂
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
import dotenv
import time
import json


def _log_usage(model, usage):
    """记录每次 API 调用的 token 消耗（成本指标）→ results/token_usage.jsonl
    reasoning_tokens（思考部分）单独记录：DeepSeek 顶层字段 / OpenAI 在 completion_tokens_details 里"""
    try:
        r = getattr(usage, "reasoning_tokens", 0)
        if not r:
            det = getattr(usage, "completion_tokens_details", None)
            r = getattr(det, "reasoning_tokens", 0) if det else 0
        p = os.path.join(os.path.dirname(__file__), "results", "token_usage.jsonl")
        rec = {"ts": time.time(), "model": model,
               "prompt_tokens": getattr(usage, "prompt_tokens", 0),
               "completion_tokens": getattr(usage, "completion_tokens", 0),
               "reasoning_tokens": r,
               "total_tokens": getattr(usage, "total_tokens", 0)}
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class chat_memory:
    def __init__(self):
        self.memory = []

    def add_message(self, role, content):
        self.memory.append({"role": role, "content": content})

    def clear_memory(self):
        self.memory = []


# provider 配置
PROVIDERS = {
    "ikun": {
        "base_url": "https://api.ikuncode.cc/v1",
        "env_key": "IKUN_API_KEY",
        "default_model": "gpt-5.6-sol",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-pro",
    },
}


class LLMClient:
    """支持多个 provider 的客户端。model 用 "provider/model" 或 "provider:model" 指定。"""

    def __init__(self, model=None, temperature=0.0, max_tokens=40000, api_key=None):
        dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
        # 解析 model 指定形式: "ikun/gpt-5.6-sol" 或 "gpt-5.6-sol"（默认 ikun）
        if model and "/" in model:
            self.provider, self.model = model.split("/", 1)
        else:
            self.provider = "ikun"
            self.model = model or PROVIDERS["ikun"]["default_model"]
        if self.provider not in PROVIDERS:
            raise ValueError(f"未知 provider: {self.provider}，可选: {list(PROVIDERS)}")

        p = PROVIDERS[self.provider]
        self.api_key = api_key or os.getenv(p["env_key"])
        if not self.api_key:
            raise ValueError(f"缺少 {p['env_key']}（provider: {self.provider}）")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=p["base_url"],
            timeout=300.0,  # pro 推理模型首 token 延迟可达数分钟（MMLU-Pro 部分题实测 120s+ 仍在思考）
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def full_model(self):
        return f"{self.provider}/{self.model}"

    def generate(self, messages, temperature=None, max_tokens=None, max_retries=50, thinking=None, effort=None):
        """单次生成。messages: list[dict]
        thinking=True/False: 思考模式开关（DeepSeek V4）；effort: low/high/max"""
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens
        last_error = None
        for attempt in range(max_retries):
            try:
                kwargs = {}
                if thinking is not None:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
                    if effort:
                        kwargs["reasoning_effort"] = effort
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mtok,
                    stream=False,
                    **kwargs,
                )
                content = resp.choices[0].message.content
                # 记录 token 消耗（成本指标）
                try:
                    _u = resp.usage
                    if _u:
                        _log_usage(self.full_model, _u)
                except Exception:
                    pass
                # DeepSeek 偶发返回空 content：持续重试直到有响应（用户要求保证一定有响应）
                if not content:
                    if attempt >= max_retries - 1:
                        last_error = RuntimeError(f"empty response from {self.model}")
                        break
                    print(f"  [LLM重试] 空响应，1s 后重试 (attempt {attempt+1}/{max_retries})...")
                    time.sleep(1)
                    continue
                return content
            except (APITimeoutError, APIConnectionError) as e:
                last_error = e
                # 超时/断连：最多重试 5 次（50 次 × 30s 会卡死整批任务）
                if attempt >= 4:
                    break
                wait = min(2**attempt, 8)
                print(f"  [LLM重试] {type(e).__name__}，{wait}s 后重试 (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            except APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    if attempt >= 4:
                        break
                    wait = min(2**attempt, 8)
                    print(f"  [LLM重试] HTTP {e.status_code}，{wait}s 后重试 (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise last_error

    def generate_response(self, chat_history, temperature=None):
        """兼容旧接口：chat_memory → str"""
        return self.generate(chat_history.memory, temperature=temperature)

    def sample_n(self, prompt, n=5, temperature=1.0, max_tokens=None):
        """对同一 prompt 采样 N 次，返回 N 个回答列表（幻觉检测核心）。
        高温采样保证多样性；语义一致性 → 置信度信号。
        """
        messages = [{"role": "user", "content": prompt}]
        out = []
        for i in range(n):
            out.append(self.generate(messages, temperature=temperature, max_tokens=max_tokens))
        return out


def get_client(model=None, temperature=0.0, max_tokens=4096):
    """快捷工厂"""
    return LLMClient(model=model, temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else None
    c = get_client(model)
    print(f"模型: {c.full_model}")
    r = c.generate([{"role": "user", "content": "南京大学在哪个城市？一句话回答。"}])
    print("回答:", r)
