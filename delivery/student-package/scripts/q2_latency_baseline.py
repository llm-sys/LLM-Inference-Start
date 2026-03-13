import argparse
import json
import statistics
import time
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in text.split("\n<END_OF_PROMPT>\n") if chunk.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/ceph/arknet/hf_models/Qwen/Qwen3-14B")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[1] / "data" / "q2_latency_prompts.txt"),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--disable-eager", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompts = read_prompts(Path(args.dataset))
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.disable_eager,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )

    latencies = []
    output_token_counts = []
    for prompt in prompts:
        started = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        output_token_counts.append(
            len(outputs[0].outputs[0].token_ids) if outputs and outputs[0].outputs else 0
        )

    result = {
        "task": "q2_latency",
        "num_prompts": len(prompts),
        "mean_latency_sec": statistics.mean(latencies),
        "median_latency_sec": statistics.median(latencies),
        "p95_latency_sec": percentile(latencies, 95),
        "max_latency_sec": max(latencies),
        "mean_output_tokens": statistics.mean(output_token_counts),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
