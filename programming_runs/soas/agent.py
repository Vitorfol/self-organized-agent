from typing import List, Dict, Union
#from api import api_chat_completions
from . import api
from . import prompt
from . import py_utils
test_generator = py_utils.MyPythonExecute("gpt-3.5-turbo-1106")



# Helper functions (not provided in the pseudocode)
def generate_skeleton(docstrings: str, model_name: str, model_kwargs: dict = None) -> str:
    # Generate skeleton code based on docstrings and unit tests
    # Ask model to respond with code only to increase chance of parsable output
    instr = "Respond ONLY with valid Python code inside a ```python``` code block. Do NOT include any explanation."
    full_prompt = instr + "\n" + prompt.PROMPT_TEMPLATES["generate_skeleton"].replace("{function}", docstrings)
    code = api.api_chat_completions(full_prompt, model_name=model_name, **(model_kwargs or {}))
    # try to extract python source; if parsing fails, try to extract code fences or def blocks
    try:
        return py_utils.extract_python_source_code_from_text(code)
    except Exception:
        # try to find ```python``` fenced block
        import re
        m = re.search(r"```python\n([\s\S]*?)```", code, re.IGNORECASE)
        if m:
            return py_utils.extract_python_source_code_from_text(m.group(1))
        # try to find first def ... block
        m2 = re.search(r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\):[\s\S]*)", code)
        if m2:
            # take until two consecutive newlines or end
            candidate = m2.group(1).split('\n\n')[0]
            return candidate
        # fallback: return raw code string so caller can handle it
        return code


def generate_code(docstrings: str, model_name: str, model_kwargs: dict = None) -> str:
    # Generate code based on docstrings and unit tests
    instr = "Respond ONLY with valid Python code inside a ```python``` code block. Do NOT include any explanation."
    full_prompt = instr + "\n" + prompt.PROMPT_TEMPLATES["generate_code"].replace("{function}", docstrings)
    code = api.api_chat_completions(full_prompt, model_name=model_name, **(model_kwargs or {}))
    try:
        return py_utils.extract_python_source_code_from_text(code)
    except Exception:
        import re
        m = re.search(r"```python\n([\s\S]*?)```", code, re.IGNORECASE)
        if m:
            return py_utils.extract_python_source_code_from_text(m.group(1))
        m2 = re.search(r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\):[\s\S]*)", code)
        if m2:
            candidate = m2.group(1).split('\n\n')[0]
            return candidate
        return code


def get_subtasks(code: str, main_func_name: str, n_gen_tests: int) -> List[tuple]:
    # Extract subtask docstrings and unit tests from the code
    # sanitize the provided code to remove SDK metadata that some models
    # (especially GPT-5 family) may include in the response. This prevents
    # ast.parse from failing on lines like `gpt-5-mini-2025-08-07`.
    import re
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

    sanitized = _sanitize(code)
    function_names = py_utils.extract_python_function_names(sanitized)

    subtasks = []
    for func_name in function_names:
        if func_name != main_func_name:
            subtask_docstrings = py_utils.extract_python_function_from_text(code, func_name)
            #subtask_unit_tests = test_generator.generate_unit_tests(subtask_docstrings, n_gen_tests)
            subtasks.append([func_name, subtask_docstrings])
    return subtasks


def update_code(code: str, test_result: str, feedback: str, model_name: str, model_kwargs: dict = None) -> str:
    # Update code based on feedback
    full_prompt = prompt.PROMPT_TEMPLATES["modify_code"].replace("{code}", code)
    full_prompt = full_prompt.replace("{test_result}", test_result)
    full_prompt = full_prompt.replace("{feedback}", feedback)
    updated_code = api.api_chat_completions(full_prompt, model_name=model_name, **(model_kwargs or {}))
    return py_utils.extract_python_source_code_from_text(updated_code)


def generate_feedback(old_code: str, test_result: bool, upper_agent_observation: Union[Dict, None], model_name: str, model_kwargs: dict = None) -> str:
    # Generate feedback based on test results and upper agent observation
    if upper_agent_observation is None:
        full_prompt = prompt.PROMPT_TEMPLATES["generate_feedback_for_root_mother"].replace("{code}", old_code)
        full_prompt = full_prompt.replace("{test_result}", test_result)
    else:
        full_prompt = prompt.PROMPT_TEMPLATES["generate_feedback"].replace("{code}", old_code)
        full_prompt = full_prompt.replace("{test_result}", test_result)
        full_prompt = full_prompt.replace("{mother_feedback}", upper_agent_observation['feedback'])
        full_prompt = full_prompt.replace("{mother_old_code}", upper_agent_observation['old_code'])
        full_prompt = full_prompt.replace("{mother_new_code}", upper_agent_observation['new_code'])
    feedback = api.api_chat_completions(full_prompt, model_name=model_name, **(model_kwargs or {}))
    return feedback



def generate_unit_tests(code: str, n_tests: int, upper_agent_observation: Union[Dict, None], model_name: str, model_kwargs: dict = None) -> List[str]:
    # Generate unit tests based on the docstrings, mother agent's feedback, old code, and new code
    full_prompt = prompt.PROMPT_TEMPLATES["generate_unit_tests"].replace("{code}", code)
    full_prompt = full_prompt.replace("{mother_feedback}", upper_agent_observation['feedback'])
    full_prompt = full_prompt.replace("{mother_test_result}", upper_agent_observation['test_result'])
    full_prompt = full_prompt.replace("{mother_old_code}", upper_agent_observation['old_code'])
    full_prompt = full_prompt.replace("{mother_new_code}", upper_agent_observation['new_code'])

    generated_tests = api.api_chat_completions(full_prompt, model_name=model_name, **(model_kwargs or {}))
    
    # Extract individual test cases from the generated content
    test_cases = generated_tests.strip().split("\n")
    
    # Return the desired number of test cases
    return test_cases[:n_tests]


class Agent:
    def __init__(self, func_name: str, docstrings: str, model_name: str, model_kwargs: dict = None):
        self.func_name = func_name
        self.docstrings = docstrings
        self.code = docstrings
        self.unit_tests = []
        self.subagents = []
        self.n_gen_tests = 1
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    def update_memory(self, code):
        self.code = code

    def update_code(self, code: str, test_result: str, feedback: str) -> str:
        # Update code based on feedback
        return update_code(code, test_result, feedback, self.model_name, model_kwargs=self.model_kwargs)
    
    def generate_feedback(self, old_code: str, test_result: bool, upper_agent_observation: Union[Dict, None]) -> str:
        return generate_feedback(old_code, test_result, upper_agent_observation, self.model_name, model_kwargs=self.model_kwargs)
    
    def generate_unit_tests(self, code: str, n_tests: int, upper_agent_observation: Union[Dict, None]) -> List[str]:
        return generate_unit_tests(code, n_tests, upper_agent_observation, self.model_name, model_kwargs=self.model_kwargs)


class ChildAgent(Agent):
    def __init__(self, func_name: str, docstrings: str, model_name: str, model_kwargs: dict = None):
        super().__init__(func_name, docstrings, model_name, model_kwargs=model_kwargs)
        self.agent_type = "child"

    def generate_code(self, docstrings: str) -> str:
        return generate_code(docstrings, self.model_name, model_kwargs=self.model_kwargs)


class MotherAgent(Agent):
    def __init__(self, func_name: str, docstrings: str, model_name: str, model_kwargs: dict = None):
        super().__init__(func_name, docstrings, model_name, model_kwargs=model_kwargs)
        self.agent_type = "mother"

    def generate_skeleton(self, docstrings: str) -> str:
        return generate_skeleton(docstrings, self.model_name, model_kwargs=self.model_kwargs)

    def get_subtasks(self, skeleton) -> List[tuple]:
        return get_subtasks(skeleton, self.func_name, self.n_gen_tests)
    




