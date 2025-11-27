from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
#from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List
import soas
import os


def run_soa(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    max_depth: int,
    is_leetcode: bool = False,
    model_params: dict = None,
) -> None:
    #exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    # Try to use the same model (provided via CLI) to generate internal tests.
    # If the account/project doesn't have access to that model (PermissionDenied),
    # fall back to a widely-available model (gpt-3.5-turbo-1106) to keep the
    # pipeline running.
    model_kwargs = model_params or {}
    try:
        model_for_gen_test = model_factory(model_name, model_kwargs=model_kwargs)
    except Exception:
        try:
            model_for_gen_test = model_factory("gpt-3.5-turbo-1106")
        except Exception:
            # last-resort: create a generic chat wrapper
            from generators.model import GPTChat
            model_for_gen_test = GPTChat("gpt-3.5-turbo-1106")
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path):
        cur_func_impl_is_None = True
        while cur_func_impl_is_None:
            cur_pass = 0
            is_solved = False
            #implementations = []
            test_feedback = []
            cur_func_impl = ""

            while cur_pass < pass_at_k and not is_solved:
                assert not is_leetcode
                tests_i = gen.internal_tests(item["prompt"], model_for_gen_test, 3)

                docstrings = item["prompt"]
                function_name = item["entry_point"]
                unit_tests = tests_i

                # If the dataset doesn't provide an external `test` (e.g. CLI inline
                # prompt), create a simple `def check(candidate): ...` wrapper from
                # the internally-generated asserts so `PyExecutor.evaluate` can run
                # the final test. The internal tests are typically asserts that
                # call the function by name, so replace calls to the function name
                # with `candidate(...)` inside the check wrapper.
                eval_test = item.get('test', '')
                if not eval_test or not eval_test.strip():
                    # build check(candidate) wrapper from unit_tests (list of assert strings)
                    try:
                        lines = []
                        for t in unit_tests:
                            # replace occurrences like `fname(` with `candidate(`
                            if isinstance(t, str):
                                lines.append(t.replace(f"{function_name}(", "candidate("))
                        if lines:
                            eval_test = "def check(candidate):\n" + "\n".join([f"    {ln}" for ln in lines])
                        else:
                            eval_test = ""
                    except Exception:
                        eval_test = ""

                max_iterations = max_iters
                result = soas.generate_and_modify_code_with_soa(function_name, docstrings, unit_tests, max_depth, max_iterations, model_name, model_kwargs=model_kwargs)
                # support old return (string) and new return (dict)
                if isinstance(result, dict):
                    impl = result.get("implementation") or ""
                    raw = result.get("raw_skeleton") or ""
                    # prefer implementation but fall back to raw skeleton
                    cur_func_impl = impl if impl and impl.strip() else raw
                else:
                    cur_func_impl = result

                # run final test; prefer external dataset `test` if present,
                # otherwise use the `eval_test` wrapper we just constructed
                is_solved = soas.final_test(function_name, cur_func_impl, eval_test)

                cur_pass += 1

                num_success += int(is_solved)

                if cur_func_impl is None:
                    cur_func_impl_is_None = True
                    break
                else:
                    cur_func_impl_is_None = False                

                item["is_solved"] = is_solved
        item["test_feedback"] = test_feedback
        # persist raw model output (if available) to help debugging extraction issues
        try:
            if isinstance(result, dict) and result.get("raw_skeleton"):
                item["raw_skeleton"] = result.get("raw_skeleton")
            else:
                item["raw_skeleton"] = ""
        except Exception:
            item["raw_skeleton"] = ""
        # ensure solution is a string; if empty, write explanatory placeholder
        if cur_func_impl is None or (isinstance(cur_func_impl, str) and not cur_func_impl.strip()):
            item["solution"] = ""  # keep empty but file will contain raw skeleton or note
        else:
            item["solution"] = cur_func_impl
        # write the solution to a separate file for easier inspection
        try:
            solutions_dir = os.path.join(os.path.dirname(log_path), "solutions")
            if not os.path.exists(solutions_dir):
                os.makedirs(solutions_dir)
            # safe filename: use item name or entry_point
            base_name = item.get("name") or item.get("entry_point") or "solution"
            safe_base = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
            entry = item.get("entry_point") or "impl"
            filename = f"{safe_base}_{entry}.py"
            filepath = os.path.join(solutions_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# Generated solution from SOA run\n")
                # if cur_func_impl is empty, try to save a helpful message with the raw skeleton
                if cur_func_impl is None or (isinstance(cur_func_impl, str) and not cur_func_impl.strip()):
                    # if we have a raw skeleton from the earlier result, prefer that
                    if isinstance(result, dict) and result.get("raw_skeleton"):
                        f.write(result.get("raw_skeleton"))
                    else:
                        f.write("# No solution generated by the model. Check the JSONL log for details.\n")
                else:
                    f.write(cur_func_impl)
        except Exception as e:
            # don't crash the run if saving fails; just print a warning
            try:
                print(f"Warning: could not write solution file: {e}")
            except Exception:
                pass
        write_jsonl(log_path, [item], append=True)
        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
        
        break

