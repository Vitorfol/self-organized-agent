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
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    payload.update(kwargs)

    response = client.chat.completions.create(**payload)

    # best-effort extraction of content
    if hasattr(response, "choices") and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content
        elif hasattr(choice, "text"):
            return choice.text
        else:
            return str(choice)
    return str(response)


def api_chat_completions(content, json_mode=False, model_name="gpt-3.5-turbo-1106", 
                         max_tokens=1000, temperature=0.0, **kwargs):
    return chat_completions(content, model=model_name, max_tokens=max_tokens, 
                                temperature=temperature, json_mode=json_mode, **kwargs)
