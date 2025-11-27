from typing import List, Union, Optional, Literal
import dataclasses

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])



import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Prefer the standard env var, fallback to the older name for compatibility
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_SECRET_KEY")
if api_key is None:
    logger.warning("No OpenAI API key found in OPENAI_API_KEY or OPENAI_API_SECRET_KEY; API calls will fail until you set one.")

# Allow configuring API base (for enterprise / proxies) via env too
api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_URL")

client_kwargs = {"api_key": api_key, "max_retries": 4}
if api_base:
    client_kwargs["base_url"] = api_base

client = OpenAI(**{k: v for k, v in client_kwargs.items() if v is not None})

#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completions(messages=None, model=None, max_tokens=None, temperature=None, json_mode=False, **kwargs):
    """Wrapper around the OpenAI client's chat completions that forwards unknown
    keyword args (so we remain compatible with newer model parameters).

    messages: list of dict or None. If None and kwargs contains 'messages', will use that.
    """
    payload = {}
    if model is not None:
        payload["model"] = model
    if messages is not None:
        payload["messages"] = messages
    # allow callers to pass content instead of messages
    if "content" in kwargs and payload.get("messages") is None:
        payload["messages"] = [{"role": "user", "content": kwargs.pop("content")}]
    if max_tokens is not None:
        # Newer models (e.g. GPT-5 family) expect `max_completion_tokens` instead
        # of the legacy `max_tokens`. Detect by model name and map accordingly.
        model_name_for_map = payload.get("model") if isinstance(payload.get("model"), str) else ""
        if model_name_for_map and ("gpt-5" in model_name_for_map or model_name_for_map.startswith("gpt5") or model_name_for_map.startswith("gpt-5")):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    # merge any other kwargs (e.g., response_format, modalities, etc.)
    payload.update(kwargs)

    # Some newer models (GPT-5 family) may restrict allowed parameter values
    model_name_for_map = payload.get("model") if isinstance(payload.get("model"), str) else ""
    is_gpt5 = bool(model_name_for_map and ("gpt-5" in model_name_for_map or model_name_for_map.startswith("gpt5") or model_name_for_map.startswith("gpt-5")))

    # Try the request; if the model rejects a parameter (e.g., temperature),
    # attempt a conservative retry (temperature=1) for GPT-5 models.
    try:
        response = client.chat.completions.create(**payload)
    except Exception as e:
        msg = str(e)
        # If GPT-5 rejects temperature (or similar), try forcing temperature=1
        if is_gpt5 and ("temperature" in msg or "Unsupported value" in msg or "does not support" in msg):
            try:
                payload["temperature"] = 1
                response = client.chat.completions.create(**payload)
            except Exception:
                # re-raise original error if retry fails
                raise
        else:
            raise

    # optional debug dump of the raw SDK response when requested
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

    # extract content in multiple possible response shapes
    def _gather_text(obj):
        out = []
        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        out.append(v)
                    else:
                        out.extend(_gather_text(v))
                return out
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    out.extend(_gather_text(v))
                return out
            if hasattr(obj, "__dict__"):
                return _gather_text(vars(obj))
            if hasattr(obj, "output"):
                return _gather_text(getattr(obj, "output"))
            if hasattr(obj, "choices"):
                return _gather_text(getattr(obj, "choices"))
        except Exception:
            return out
        if isinstance(obj, str):
            out.append(obj)
            return out
        return out

    texts = _gather_text(response)
    candidate = "\n".join(t.strip() for t in texts if isinstance(t, str) and t.strip())
    if candidate:
        # prefer fenced python blocks; sanitize obvious SDK metadata first
        import re
        def _sanitize(text: str) -> str:
            out_lines = []
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                if re.match(r'^(chatcmpl-|req_|gpt-5|gpt5|gpt-4|gpt4|length$|assistant$|chat\.completion$|default$)', s):
                    continue
                if len(s) < 80 and re.match(r'^[A-Za-z0-9_\-]{4,}$', s) and not re.search(r'[():=<>"\'"\[\]{}]', s):
                    continue
                out_lines.append(ln)
            return "\n".join(out_lines)

        candidate = _sanitize(candidate)
        m = re.search(r"```python\n([\s\S]*?)```", candidate, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r"```\n([\s\S]*?)```", candidate)
        if m2:
            return m2.group(1)
        # prefer returning from first `def ` onwards if found
        m3 = re.search(r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\):[\s\S]*)", candidate)
        if m3:
            return m3.group(1)
        return candidate
    try:
        return str(response)
    except Exception:
        return ""


def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
        **kwargs,
) -> Union[List[str], str]:
    # Accept any model name (gpt-3.5/gpt-4/gpt-5...) and forward extra kwargs.
    if not isinstance(model, str):
        raise ValueError("model must be a string")

    messages = [{"role": "user", "content": prompt}]
    response = chat_completions(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **({} if stop_strs is None else {"stop": stop_strs}),
        **kwargs,
    )

    # response shape may vary; handle common shapes safely
    if hasattr(response, "choices") and len(response.choices) > 0:
        choice = response.choices[0]
        # new chat responses put text in .message.content; older completions used .text
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            out = choice.message.content
        elif hasattr(choice, "text"):
            out = choice.text
        else:
            out = str(choice)
    else:
        out = str(response)

    if num_comps == 1:
        return out

    # return list of outputs
    outs = []
    for c in getattr(response, "choices", []):
        if hasattr(c, "message") and hasattr(c.message, "content"):
            outs.append(c.message.content)
        elif hasattr(c, "text"):
            outs.append(c.text)
        else:
            outs.append(str(c))
    return outs
#--------------------------------------------------------------

#@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    # don't restrict model names; accept gpt-3.5, gpt-4, gpt-5, etc.
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    #print([dataclasses.asdict(message) for message in messages])
    #response = openai.ChatCompletion.create(
    #    model=model,
    #    messages=[dataclasses.asdict(message) for message in messages],
    #    max_tokens=max_tokens,
    #    temperature=temperature,
    #    top_p=1,
    #    frequency_penalty=0.0,
    #    presence_penalty=0.0,
    #    n=num_comps,
    #)
    response = chat_completions(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # chat_completions may already return an extracted string (we made the
    # wrapper robust to different SDK shapes). If so, just return it.
    if isinstance(response, str):
        if num_comps == 1:
            return response
        else:
            # best-effort: duplicate the single response to satisfy callers
            return [response for _ in range(num_comps)]

    # otherwise response is likely an SDK response object with choices
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False
        # container for model-specific kwargs provided at construction time
        self.model_kwargs = {}

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1, **model_kwargs) -> Union[List[str], str]:
        # merge model instance default kwargs with call-time kwargs
        merged = {}
        if hasattr(self, "model_kwargs") and isinstance(self.model_kwargs, dict):
            merged.update(self.model_kwargs)
        merged.update(model_kwargs)
        # Avoid passing duplicate keyword args: let merged override defaults
        mk = dict(merged)  # copy
        mk_max_tokens = mk.pop("max_tokens", max_tokens)
        mk_temperature = mk.pop("temperature", temperature)
        mk_num_comps = mk.pop("num_comps", num_comps)
        # forward model-specific kwargs (remaining keys) to the chat function
        return gpt_chat(self.name, messages, max_tokens=mk_max_tokens, temperature=mk_temperature, num_comps=mk_num_comps, **mk)


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")


class GPT35(GPTChat):
    def __init__(self):
        #super().__init__("gpt-3.5-turbo")
        super().__init__("gpt-3.5-turbo-1106")


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]

        return out


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out
