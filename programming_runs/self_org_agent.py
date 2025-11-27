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

                max_iterations = max_iters
                cur_func_impl = soas.generate_and_modify_code_with_soa(function_name, docstrings, unit_tests, max_depth, max_iterations, model_name, model_kwargs=model_kwargs)
                eval_test = item['test']
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
                f.write(cur_func_impl if cur_func_impl is not None else "# No solution generated\n")
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

