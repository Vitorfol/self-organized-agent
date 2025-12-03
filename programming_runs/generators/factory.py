from .py_generate import PyGenerator
#from .rs_generate import RsGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci, GPTChat


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        assert False
        #return RsGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str, model_kwargs: dict = None) -> ModelBase:
    """Create a model instance based on the model name.
    
    Supports:
    - GPT-4 family: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, etc. (uses max_tokens)
    - GPT-3.5 family: gpt-3.5-turbo, gpt-3.5-turbo-1106, etc. (uses max_tokens)
    - GPT-5 family: gpt-5, gpt-5-mini, gpt-5-turbo, etc. (uses max_completion_tokens)
    - Legacy models: text-davinci-*, codellama, starchat
    """
    model_lower = model_name.lower()
    
    # GPT-4 family (all variants)
    if "gpt-4" in model_lower or model_lower.startswith("gpt4"):
        m = GPT4() if model_name == "gpt-4" else GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
    # GPT-3.5 family
    elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
        m = GPT35() if model_name == "gpt-3.5-turbo-1106" else GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
    # GPT-5 family (new models)
    elif "gpt-5" in model_lower or model_lower.startswith("gpt5"):
        m = GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
    # Generic GPT models (catch-all for other gpt-* variants)
    elif model_name.startswith("gpt-") or model_name.startswith("gpt"):
        m = GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
    # Legacy models
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
