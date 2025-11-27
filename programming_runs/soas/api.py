import os
import logging
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

# client initialization: prefer OPENAI_API_KEY, fall back to OPENAI_API_SECRET_KEY
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_SECRET_KEY")
if api_key is None:
    logger.warning("No OpenAI API key found (OPENAI_API_KEY / OPENAI_API_SECRET_KEY). API calls will fail until you set one.")

client_kwargs = {"api_key": api_key, "max_retries": 4}
api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_URL")
if api_base:
    client_kwargs["base_url"] = api_base

client = OpenAI(**{k: v for k, v in client_kwargs.items() if v is not None})


#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completions(content, model, max_tokens=1000, temperature=0.0, json_mode=False, **kwargs):
    """Wrapper for chat completions. Accepts extra kwargs and forwards them to the client.

    content: string to be sent as a single-user message.
    model: model name string.
    Additional kwargs are forwarded (e.g., response_format, modalities, etc.).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    # map legacy `max_tokens` to `max_completion_tokens` for newer models (GPT-5 family)
    if max_tokens is not None:
        if isinstance(model, str) and ("gpt-5" in model or model.startswith("gpt5") or model.startswith("gpt-5")):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
    payload.update(kwargs)

    # Normalize for GPT-5 family models: enforce allowed parameter values
    is_gpt5 = isinstance(model, str) and ("gpt-5" in model or model.startswith("gpt5") or model.startswith("gpt-5"))

    try:
        response = client.chat.completions.create(**payload)
    except Exception as e:
        msg = str(e)
        if is_gpt5 and ("temperature" in msg or "Unsupported value" in msg or "does not support" in msg):
            # retry with conservative temperature
            try:
                payload["temperature"] = 1
                response = client.chat.completions.create(**payload)
            except Exception:
                raise
        else:
            raise

    # best-effort extraction of content
    # optional debug dump
    try:
        import time
        if os.getenv("DUMP_OPENAI_RESPONSE") == "1":
            dump_path = os.path.join(os.getcwd(), "root", f"debug_response_{int(time.time())}.txt")
            try:
                with open(dump_path, "w", encoding="utf-8") as df:
                    df.write("PAYLOAD:\n")
                    df.write(str(payload) + "\n\n")
                    df.write("RESPONSE_REPR:\n")
                    df.write(repr(response))
            except Exception:
                pass
    except Exception:
        pass

    def _gather_text(obj):
        """Recursively walk the response and collect any string leaves or 'text' fields."""
        out = []
        try:
            # OpenAI SDK objects often expose attributes; try dict-like first
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        out.append(v)
                    else:
                        out.extend(_gather_text(v))
                return out
            # list/tuple
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    out.extend(_gather_text(v))
                return out
            # objects with __dict__
            if hasattr(obj, "__dict__"):
                return _gather_text(vars(obj))
            # fallback: try attribute access for common names
            if hasattr(obj, "output"):
                return _gather_text(getattr(obj, "output"))
            if hasattr(obj, "choices"):
                return _gather_text(getattr(obj, "choices"))
        except Exception:
            return out
        # string leaf
        if isinstance(obj, str):
            out.append(obj)
            return out
        return out

    texts = _gather_text(response)
    candidate = "\n".join(t.strip() for t in texts if isinstance(t, str) and t.strip())
    if candidate:
        # prefer fenced python blocks if present
        import re
        # sanitize candidate by removing common SDK metadata lines
        def _sanitize(text: str) -> str:
            out_lines = []
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                # skip obvious SDK metadata tokens/headers
                if re.match(r'^(chatcmpl-|req_|gpt-5|gpt5|gpt-4|gpt4|length$|assistant$|chat\.completion$|default$)', s):
                    continue
                # skip short single-token lines that look like ids/hashes
                if len(s) < 80 and re.match(r'^[A-Za-z0-9_\-]{4,}$', s) and not re.search(r'[():=<>"\'"\[\]{}]', s):
                    continue
                out_lines.append(ln)
            return "\n".join(out_lines)

        candidate = _sanitize(candidate)
        m = re.search(r"```python\n([\s\S]*?)```", candidate, re.IGNORECASE)
        if m:
            return m.group(1)
        # plain fenced block
        m2 = re.search(r"```\n([\s\S]*?)```", candidate)
        if m2:
            return m2.group(1)
        # fallback: return from first `def ` occurrence to the end (best-effort)
        m3 = re.search(r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\):[\s\S]*)", candidate)
        if m3:
            return m3.group(1)
        return candidate
    # last resort: return str(response)
    try:
        return str(response)
    except Exception:
        return ""


def api_chat_completions(content, json_mode=False, model_name="gpt-3.5-turbo-1106", 
                         max_tokens=1000, temperature=0.0, **kwargs):
    return chat_completions(content, model=model_name, max_tokens=max_tokens, 
                                temperature=temperature, json_mode=json_mode, **kwargs)
