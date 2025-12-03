from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
#from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List
import time
import soas
import os
import concurrent.futures
from solution_validator import validate_implementation, is_code_empty


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
    timeout: int = 0,
    max_turns: int = 1,
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
    # overall timeout handling
    start_time = time.time()
    deadline = None
    try:
        t = int(timeout)
        if t and t > 0:
            deadline = start_time + t
    except Exception:
        deadline = None

    for i, item in enumerate_resume(dataset, log_path):
        # check overall timeout before starting the next item
        if deadline is not None and time.time() >= deadline:
            print_v(f"timeout reached before item {i+1}/{num_items}; stopping run")
            break

        # Simplified flow with controlled turns: generate up to `max_turns`
        # attempts per item, or stop earlier if we get a non-empty implementation.
        test_feedback = []

        docstrings = item["prompt"]
        function_name = item["entry_point"]

        cur_func_impl = None
        attempts = 0
        timed_out = False
        while attempts < max_turns:
            # check timeout inside the per-item loop
            if deadline is not None and time.time() >= deadline:
                print_v(f"timeout reached during item {i+1}/{num_items} after {attempts} attempts")
                break

            # compute remaining time (seconds) for this attempt
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.time())

            # Run the generation in a worker and respect the remaining time
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        soas.generate_and_modify_code_with_soa,
                        function_name,
                        docstrings,
                        [],
                        max_depth,
                        max_iters,  # Now using the max_iters parameter from command line
                        model_name,
                        model_kwargs,
                    )
                    # If remaining is None (no deadline) wait indefinitely, else wait up to remaining
                    if remaining is None or remaining <= 0:
                        result = future.result()
                    else:
                        result = future.result(timeout=remaining)
            except concurrent.futures.TimeoutError:
                print_v(f"generation timed out for item {i+1}/{num_items} after {attempts} attempts (deadline reached)")
                timed_out = True
                # we did not get a result for this attempt
                result = None
            except Exception as e:
                # propagate other exceptions as non-fatal for the run; capture and continue
                print_v(f"generation error for item {i+1}/{num_items}: {e}")
                result = None
            # support old return (string) and new return (dict)
            if isinstance(result, dict):
                impl = result.get("implementation") or ""
                raw = result.get("raw_skeleton") or ""
                # Use raw_skeleton as the complete code (includes all functions)
                # implementation is just the combined agent codes which may be incomplete
                candidate_impl = raw if raw and raw.strip() else impl
                print_v(f"  Implementation length: {len(impl)}, Raw skeleton length: {len(raw)}")
                print_v(f"  Using: {'raw_skeleton' if raw and raw.strip() else 'implementation'}")
            else:
                candidate_impl = result

            if candidate_impl is not None and isinstance(candidate_impl, str) and candidate_impl.strip():
                cur_func_impl = candidate_impl
                break

            attempts += 1

        if timed_out:
            # stop the entire run if we hit the overall deadline
            print_v("overall deadline reached; terminating run")
            break

        # Validate the generated implementation
        is_valid = False
        validation_issues = []
        
        if cur_func_impl is not None and isinstance(cur_func_impl, str) and cur_func_impl.strip():
            # Check if code is actually implemented (not just pass statements)
            is_valid, validation_issues = validate_implementation(cur_func_impl, strict=False)
            
            if not is_valid:
                print_v(f"⚠️  Generated code has issues:")
                for issue in validation_issues:
                    print_v(f"    {issue}")
            else:
                print_v(f"✓ Code validation passed")
        else:
            validation_issues = ["No implementation generated"]
            print_v(f"⚠️  No implementation generated")
        
        # Mark as solved only if validation passed
        item["is_solved"] = is_valid
        item["test_feedback"] = test_feedback
        item["validation_issues"] = validation_issues if not is_valid else []
        
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
                print_v(f"warning: failed to save solution file: {e}")
            except Exception:
                pass
        
        # Update success count
        if is_valid:
            num_success += 1
        
        write_jsonl(log_path, [item], append=True)
        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')

