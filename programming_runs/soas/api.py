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

# Store the last raw SDK response for debugging and fallback when extraction fails.
LAST_RAW_RESPONSE = None
LAST_RAW_RESPONSE_REPR = ""


def is_gpt5_model(model_name: str) -> bool:
    """Detect if a model is from the GPT-5 family.
    
    GPT-5 models use max_completion_tokens instead of max_tokens.
    """
    if not isinstance(model_name, str):
        return False
    model_lower = model_name.lower()
    # Check for gpt-5 or gpt5 patterns
    return "gpt-5" in model_lower or model_lower.startswith("gpt5")


def is_gpt4_or_older(model_name: str) -> bool:
    """Detect if a model is GPT-4, GPT-3.5, or older.
    
    These models use max_tokens parameter.
    """
    if not isinstance(model_name, str):
        return False
    model_lower = model_name.lower()
    # Check for gpt-4, gpt-3.5, gpt-3, davinci, etc.
    return (
        "gpt-4" in model_lower or 
        "gpt-3" in model_lower or
        model_lower.startswith("gpt4") or
        model_lower.startswith("gpt3") or
        "davinci" in model_lower or
        "curie" in model_lower or
        "babbage" in model_lower or
        "ada" in model_lower
    )


#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completions(content, model, max_tokens=4096, temperature=0.0, json_mode=False, **kwargs):
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
    
    # Handle max_tokens / max_completion_tokens intelligently
    # Check if user passed either parameter in kwargs
    user_max_completion_tokens = kwargs.pop("max_completion_tokens", None)
    user_max_tokens = kwargs.pop("max_tokens", None)
    
    is_gpt5 = is_gpt5_model(model)
    
    # Determine which token limit to use based on model and what user provided
    if is_gpt5:
        # GPT-5 uses max_completion_tokens
        token_limit = user_max_completion_tokens or user_max_tokens or max_tokens
        if token_limit is not None:
            payload["max_completion_tokens"] = token_limit
    else:
        # GPT-4 and older use max_tokens
        token_limit = user_max_tokens or user_max_completion_tokens or max_tokens
        if token_limit is not None:
            payload["max_tokens"] = token_limit
    
    payload.update(kwargs)

    # Normalize for GPT-5 family models: enforce allowed parameter values
    # (is_gpt5 already set above)

    # Try request; on some GPT-5 endpoints a BadRequest may indicate the
    # model output limit (max tokens) was reached. In that case, retry with
    # progressively larger max_completion_tokens values.
    try:
        response = client.chat.completions.create(**payload)
    except Exception as e:
        msg = str(e)
        # handle temperature/value rejections as before
        if is_gpt5 and ("temperature" in msg or "Unsupported value" in msg or "does not support" in msg):
            try:
                payload["temperature"] = 1
                response = client.chat.completions.create(**payload)
            except Exception:
                raise
        # handle token limit errors by retrying with higher limits
        elif is_gpt5 and ("max_tokens" in msg or "max_completion_tokens" in msg or "model output limit" in msg or "max output" in msg):
            # progressive token caps to try (conservative -> larger)
            for cap in (2048, 8192, 32768):
                try:
                    if "max_completion_tokens" in payload:
                        payload["max_completion_tokens"] = cap
                    else:
                        payload["max_tokens"] = cap
                    response = client.chat.completions.create(**payload)
                    break
                except Exception:
                    response = None
                    continue
            if response is None:
                # re-raise original error if retries exhausted
                raise
        else:
            raise

    # store raw response for debugging
    try:
        global LAST_RAW_RESPONSE, LAST_RAW_RESPONSE_REPR
        LAST_RAW_RESPONSE = response
        LAST_RAW_RESPONSE_REPR = repr(response)
    except Exception:
        pass

    # If the model finished due to length (truncated), attempt a retry with
    # larger max tokens for GPT-5 family. Some endpoints return finish_reason
    # == 'length' but don't raise an error; we should try to request more.
    try:
        def _response_truncated(resp):
            try:
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    ch = resp.choices[0]
                    # new-style may use finish_reason attr
                    if hasattr(ch, "finish_reason") and ch.finish_reason == "length":
                        return True
                    # empty assistant content is suspicious
                    if hasattr(ch, "message") and hasattr(ch.message, "content") and not ch.message.content:
                        return True
                # usage-based heuristic
                if hasattr(resp, "usage"):
                    u = resp.usage
                    try:
                        # if completion tokens equals a fairly large number, maybe truncated
                        if getattr(u, "completion_tokens", 0) >= 1000:
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
            return False

        if is_gpt5 and _response_truncated(response):
            # try progressively larger caps
            for cap in (8192, 32768, 65536):
                try:
                    if "max_completion_tokens" in payload:
                        payload["max_completion_tokens"] = cap
                    else:
                        payload["max_tokens"] = cap
                    response_retry = client.chat.completions.create(**payload)
                    # update raw and use the retry if it contains content
                    if hasattr(response_retry, "choices") and len(response_retry.choices) > 0:
                        ch = response_retry.choices[0]
                        txt = ""
                        if hasattr(ch, "message") and hasattr(ch.message, "content"):
                            txt = ch.message.content
                        elif hasattr(ch, "text"):
                            txt = ch.text
                        if txt and txt.strip():
                            response = response_retry
                            LAST_RAW_RESPONSE = response_retry
                            LAST_RAW_RESPONSE_REPR = repr(response_retry)
                            break
                except Exception:
                    continue
    except Exception:
        pass

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
        # sanitize candidate by removing common SDK metadata lines (but don't
        # aggressively strip single-token lines which may be valid code)
        def _sanitize(text: str) -> str:
            out_lines = []
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                # skip obvious SDK metadata tokens/headers
                if re.match(r'^(chatcmpl-|req_|gpt-5|gpt5|gpt-4|gpt4|length$|assistant$|chat\.completion$|default$)', s):
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
