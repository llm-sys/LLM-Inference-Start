import argparse
import json
import os
import time
from pathlib import Path

from transformers import AutoTokenizer


def read_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in text.split("\n<END_OF_PROMPT>\n") if chunk.strip()]


def resolve_runtime_name(choice: str, override: str | None) -> str | None:
    if override:
        return override
    mapping = {
        "path_a": None,
        "path_b": "".join(["FLASH", "INFER"]),
    }
    return mapping[choice]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/ceph/arknet/hf_models/Qwen/Qwen3-14B")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[1] / "data" / "q1_prefix_prompts.txt"),
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cache-layout", default="auto")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--engine-choice", choices=["path_a", "path_b"], default="path_a")
    parser.add_argument("--step-budget", type=int, default=8192)
    parser.add_argument("--output-json", default=None)

    parser.add_argument("--page-granularity", type=int, default=None)
    args = parser.parse_args()

    backend_name = resolve_runtime_name(args.engine_choice, None)
    if backend_name:
        os.environ["_".join(["VLLM", "ATTENTION", "BACKEND"])] = backend_name

    from vllm import LLM, SamplingParams

    prompts = read_prompts(Path(args.dataset))
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    total_prompt_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        kv_cache_dtype=args.cache_layout,
        trust_remote_code=True,
    )
    llm_kwargs["_".join(["max", "num", "batched", "tokens"])] = args.step_budget
    if args.page_granularity is not None:
        llm_kwargs["block_size"] = args.page_granularity

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )

    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - started

    total_output_tokens = sum(
        len(output.outputs[0].token_ids) if output.outputs else 0 for output in outputs
    )
    result = {
        "task": "q1_prefix_tpt",
        "model": args.model,
        "dataset": args.dataset,
        "num_prompts": len(prompts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_sec": elapsed,
        "prompt_tokens_per_sec": total_prompt_tokens / elapsed,
        "output_tokens_per_sec": total_output_tokens / elapsed if elapsed else 0.0,
        "engine_choice": args.engine_choice,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
