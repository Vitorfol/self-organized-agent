import os
import argparse
from test_acc import run_test_acc
from utils import read_jsonl, read_jsonl_gz
from self_org_agent import run_soa

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run", default="soa")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`", default="self-org-agent")
    # default language is python for CLI tasks
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`", default="py")
    # allow passing a single task via CLI instead of a dataset file
    parser.add_argument("--prompt", type=str,
                        help="Inline prompt (function signature/docstring) to run as a single task")
    parser.add_argument("--task_name", type=str,
                        help="Name for the inline task (used as dataset name)")
    # don't set a non-None default here so we can detect when the user
    # didn't supply an entry point and infer it from the prompt
    parser.add_argument("--entry_point", type=str,
                        help="Entry point / function name for the inline task", default=None)
    parser.add_argument("--test", type=str,
                        help="Unit test code (string) for the inline task", default=None)
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--max_depth", type=int,
                        help="The maximum depth of tree of self-organizing agents", default=2)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)

    #parser.add_argument("--is_leetcode", action='store_true',
    #                    help="To run the leetcode benchmark")  # Temporary

    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "self-org-agent":
        return kwargs_wrapper_gen(run_soa, delete_keys=["expansion_factor"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # if an inline prompt is provided, create a single-item dataset in memory
    if args.prompt:
        # support natural language prompts like "so create a binary search algorithm"
        # If the prompt already contains a function definition, keep it.
        import re
        prompt_text = args.prompt
        func_name = None
        try:
            m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", args.prompt)
            if m:
                func_name = m.group(1)
        except Exception:
            func_name = None

        # If no function signature is present, try to infer a friendly name from the NL prompt
        if not func_name:
            s = args.prompt.lower()
            # some simple heuristics for common tasks
            if re.search(r"\bbinary search\b", s):
                inferred = "binary_search"
                prompt_text = f"def {inferred}(arr: list[int], target: int) -> int:\n    \"\"\"{args.prompt.strip()}\n    \"\"\"\n"
            elif re.search(r"\bfib(?:onacci)?\b", s):
                inferred = "fib"
                prompt_text = f"def {inferred}(n: int) -> int:\n    \"\"\"{args.prompt.strip()}\n    \"\"\"\n"
            else:
                words = re.findall(r"[a-z0-9]+", s)
                if words:
                    inferred = "_".join(words[:3])
                else:
                    inferred = "cli_task"
                if inferred[0].isdigit():
                    inferred = "task_" + inferred
                # generic signature when we don't know the exact args
                prompt_text = f"def {inferred}(*args, **kwargs):\n    \"\"\"{args.prompt.strip()}\n    \"\"\"\n"
            func_name = inferred

        entry_point = args.entry_point if args.entry_point else (func_name if func_name else "main")
        dataset_name = args.task_name if args.task_name else (entry_point if entry_point else "cli_task")

        dataset = [{
            "name": dataset_name,
            "language": args.language,
            "prompt": prompt_text,
            "doctests": "transform",
            "entry_point": entry_point,
            "test": args.test if args.test is not None else ""
        }]
    else:
        # get the dataset name from the path
        dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset from file if not provided inline
    if not args.prompt:
        print(f'Loading the dataset from {args.dataset_path}...')
        if args.dataset_path.endswith(".jsonl"):
            dataset = read_jsonl(args.dataset_path)
        elif args.dataset_path.endswith(".jsonl.gz"):
            dataset = read_jsonl_gz(args.dataset_path)
        else:
            raise ValueError(
                f"Dataset path `{args.dataset_path}` is not supported")
    else:
        print("Using inline prompt as single-task dataset")

    print(f"Loaded {len(dataset)} examples")
    # start the run
    # evaluate with pass@k
    run_strategy(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        expansion_factor=args.expansion_factor,
        max_depth=args.max_depth,
        is_leetcode=False, #args.is_leetcode
    )

    print(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)
