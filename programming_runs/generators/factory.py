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
    # support specific named classes for older behavior, but default to a
    # generic GPTChat wrapper for any `gpt-*` model (including gpt-5 names)
    if "gpt-4" in model_name:
        m = GPT4()
        m.model_kwargs = model_kwargs or {}
        return m
    elif model_name == "gpt-3.5-turbo-1106":
        m = GPT35()
        m.model_kwargs = model_kwargs or {}
        return m
    elif model_name.startswith("gpt-") or model_name.startswith("gpt"):
        # generic chat model wrapper (supports gpt-5 family names too)
        m = GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
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
