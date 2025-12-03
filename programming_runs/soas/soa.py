from typing import List, Dict, Union, Tuple
from . import py_utils
from .agent import Agent, ChildAgent, MotherAgent
from . import api as soas_api

# Import validator to check for unimplemented functions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from solution_validator import validate_implementation
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False
    def validate_implementation(code, strict=False):
        """Fallback if validator not available."""
        return True, []


py_executor = py_utils.MyPythonExecute("gpt-3.5-turbo-1106")


def generate_agent(agent: Agent, subtask_funcname: str, subtask_docstrings: str, depth: int, max_depth: int, model_name: str, model_kwargs: dict = None):
    if depth + 1 == max_depth:
        next_agent = ChildAgent(subtask_funcname, subtask_docstrings, model_name, model_kwargs=model_kwargs)
    else:
        next_agent = MotherAgent(subtask_funcname, subtask_docstrings, model_name, model_kwargs=model_kwargs)
    agent.subagents.append(next_agent)
    return next_agent


def generate(agent: Agent, depth: int, max_depth: int, model_name: str, model_kwargs: dict = None):
    print(f"generation\tmax_depth: {max_depth}\t depth: {depth}\tagent_type: {agent.agent_type}")

    if depth == max_depth:  # Child
        code = agent.generate_code(agent.docstrings)
        agent.update_memory(code)
    else: # Mother
        skeleton = agent.generate_skeleton(agent.docstrings)
        # preserve the raw skeleton (model reply) separately so we don't lose
        # the original text when we sanitize/parse it into `code`.
        try:
            # if the extracted skeleton is empty, try to use the raw SDK
            # response repr as a fallback so we at least persist what the
            # model returned.
            if skeleton is None or not isinstance(skeleton, str) or not skeleton.strip():
                raw = getattr(soas_api, "LAST_RAW_RESPONSE_REPR", "")
                agent.raw_skeleton = raw or ""
            else:
                agent.raw_skeleton = skeleton
        except Exception:
            agent.raw_skeleton = ""
        # Try to extract the specific function implementation; if extraction
        # fails (agent returned an unexpected format), fall back to using the
        # entire python source extracted from the skeleton so the run can continue.
        try:
            code = py_utils.extract_python_function_from_text(skeleton, agent.func_name)
        except Exception:
            # fallback: try extracting source code (may contain multiple funcs)
            try:
                code = py_utils.extract_python_source_code_from_text(skeleton)
            except Exception:
                # last resort: attempt to salvage by extracting a fenced python
                # code block or the first `def` block via simple heuristics
                import re
                m = re.search(r"```python\n([\s\S]*?)```", skeleton, re.IGNORECASE)
                if m:
                    code = m.group(1)
                else:
                    m2 = re.search(r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\):[\s\S]*)", skeleton)
                    if m2:
                        candidate = m2.group(1).split('\n\n')[0]
                        code = candidate
                    else:
                        # last resort: use the raw skeleton
                        code = skeleton
        agent.update_memory(code)
        print("code (agent):")
        print(code)
        print()
        # use the sanitized `code` (extracted function/source) when deriving
        # subtasks; passing the raw `skeleton` can contain SDK metadata and
        # break the AST-based name extraction.
        for subtask_funcname, subtask_docstrings in agent.get_subtasks(code):
            next_agent = generate_agent(agent, subtask_funcname, subtask_docstrings, depth, max_depth, model_name, model_kwargs=model_kwargs)
            generate(next_agent, depth + 1, max_depth, model_name)


def modify(agent: Agent, test_result: bool, upper_agent_observation: Union[Dict, None], depth: int, max_depth: int):
    print(f"modification\tmax_depth: {max_depth}\t depth: {depth}\tagent_type: {agent.agent_type}")
    print(f"code:\n\n{agent.code}\n")
    print(f"test:\n\n{test_result}\n")

    feedback = agent.generate_feedback(agent.code, test_result, upper_agent_observation)

    old_code = agent.code
    new_code = agent.update_code(old_code, test_result, feedback)
    agent.update_memory(new_code)
    
    # Update raw_skeleton too so validation sees the latest code
    if hasattr(agent, 'raw_skeleton'):
        agent.raw_skeleton = new_code

    for subagent in agent.subagents:
        implementation = combine_implementations(subagent)
        agent_observation = {"feedback": feedback, "old_code": old_code, "new_code": new_code, "test_result": test_result}
        subagent.unit_tests = subagent.generate_unit_tests(subagent.code, subagent.n_gen_tests, agent_observation)
        is_passing, subagent_test_result = evaluate(implementation, subagent.unit_tests)
        if not is_passing:
            modify(subagent, subagent_test_result, agent_observation, depth + 1, max_depth)


def generate_and_modify_code_with_soa(function_name: str, docstrings: str, unit_tests: List[str], max_depth: int, max_iterations: int, model_name: str, model_kwargs: dict = None) -> dict:
    """Generate and iteratively modify code with SOA.

    Returns a dict with:
      - implementation: the combined implementation string (may be empty)
      - raw_skeleton: the last root mother code/skeleton (best-effort)
    """
    root_mother = MotherAgent(function_name, docstrings, model_name=model_name, model_kwargs=model_kwargs)
    generate(root_mother, 1, max_depth, model_name, model_kwargs=model_kwargs)

    for i in range(max_iterations):
        print(f"iter: {i+1}")
        final_implementation = combine_implementations(root_mother)
        
        # Use raw_skeleton for validation if available (it contains all helper functions)
        # otherwise fall back to combined implementation
        code_to_validate = getattr(root_mother, "raw_skeleton", "") or final_implementation
        
        # First check if code is actually implemented (not just pass statements)
        is_implemented, validation_issues = validate_implementation(code_to_validate, strict=False)
        
        if not is_implemented:
            print(f"  ⚠️  Code has unimplemented functions: {', '.join(validation_issues)}")
            # If this is the last iteration, stop trying
            if i + 1 >= max_iterations:
                print(f"  ⚠️  Max iterations reached without full implementation")
                break
            # Continue to next iteration to try to implement them
            modify(root_mother, "Functions not implemented. Please provide complete implementations for all functions, not just 'pass' statements.", None, 1, max_depth)
            continue
        
        # If there are no unit tests, we cannot verify correctness beyond implementation check
        # In this case, we consider it "solved" since all functions are implemented
        if not unit_tests or len(unit_tests) == 0:
            print("  ℹ️  No unit tests provided. All functions are implemented.")
            break  # Exit since we can't improve without tests
        
        # If implemented, test it
        is_passing, test_result = evaluate(final_implementation, unit_tests)
        if is_passing:
            print("solved.")
            break
        else:
            print(f"  ❌ Tests failed: {test_result[:100]}...")
            modify(root_mother, test_result, None, 1, max_depth)

    combined = combine_implementations(root_mother)
    # raw_skeleton: best-effort last known root mother code
    raw = getattr(root_mother, "raw_skeleton", "") or getattr(root_mother, "code", "") or ""
    return {"implementation": combined, "raw_skeleton": raw}


def evaluate(code: str, unit_tests: List[str]) -> Tuple[bool, str]:
    # Evaluate code using unit tests
    is_passing, test_result, _ = py_executor.exe.execute(code, unit_tests)
    return is_passing, test_result


def combine_implementations(agent: Agent) -> str:
    # Initialize an empty list to store the collected code
    code_parts = [agent.code]

    # Recursively traverse the subagents and collect their code
    def traverse_subagents(subagent: Agent):
        code_parts.append(subagent.code)
        for sub_subagent in subagent.subagents:
            traverse_subagents(sub_subagent)

    # Traverse the subagents of the current agent
    for subagent in agent.subagents:
        traverse_subagents(subagent)

    # Combine the collected code parts into a single text
    combined_code = "\n\n".join(code_parts)

    return combined_code


def final_test(function_name: str, implementation: str, unit_tests: str) -> bool:
    is_solved = py_executor.evaluate(function_name, implementation, unit_tests)
    return is_solved
