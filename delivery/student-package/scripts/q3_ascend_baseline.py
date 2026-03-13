import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import time
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in text.split("\n<END_OF_PROMPT>\n") if chunk.strip()]


def ensure_runtime_dirs(base_dir: Path) -> None:
    runtime_dirs = [
        base_dir / "home",
        base_dir / ".cache",
        base_dir / "tmp",
        base_dir / "rpc",
        base_dir / "pycache",
        base_dir / ".config" / "vllm",
        base_dir / "vllm-cache",
        base_dir / "torchinductor-cache",
        base_dir / "triton-cache",
        base_dir / "ascend-log",
    ]
    for path in runtime_dirs:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["LANG"] = "C.UTF-8"
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HOME"] = str(base_dir / "home")
    os.environ["XDG_CACHE_HOME"] = str(base_dir / ".cache")
    os.environ["TMPDIR"] = str(base_dir / "tmp")
    os.environ["PYTHONPYCACHEPREFIX"] = str(base_dir / "pycache")
    os.environ["VLLM_CACHE_ROOT"] = str(base_dir / "vllm-cache")
    os.environ["VLLM_CONFIG_ROOT"] = str(base_dir / ".config" / "vllm")
    os.environ["VLLM_RPC_BASE_PATH"] = str(base_dir / "rpc")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(base_dir / "torchinductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(base_dir / "triton-cache")
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(base_dir / "ascend-log")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_mode_token(mode_token: str) -> dict:
    config = {}
    if mode_token == "mode_b":
        config["cudagraph_mode"] = "_".join(["FULL", "DECODE", "ONLY"])
    elif mode_token != "mode_a":
        raise ValueError(f"Unsupported mode token: {mode_token}")
    return config


def resolve_layout_token(layout_token: str) -> dict:
    mapping = {
        "layout_a": {"tp_size": 2, "ep_enabled": False},
        "layout_b": {"tp_size": 2, "ep_enabled": True},
        "layout_c": {"tp_size": 1, "ep_enabled": True},
    }
    return mapping[layout_token]


def split_evenly(items: list[str], parts: int, rank: int) -> list[str]:
    floor = len(items) // parts
    remainder = len(items) % parts

    def start(idx: int) -> int:
        return idx * floor + min(idx, remainder)

    return items[start(rank) : start(rank + 1)]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
    return values[idx]


def build_compile_dict(mode_token: str, cache_dir: Path) -> dict:
    config = {"cache_dir": str(cache_dir)}
    config.update(resolve_mode_token(mode_token))
    return config


def run_layout_ab(
    *,
    model: str,
    prompts: list[str],
    layout: str,
    mode_token: str,
    runtime_dir: Path,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_tokens: int,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    settings = resolve_layout_token(layout)
    world_size = 2
    master_addr = "127.0.0.1"
    master_port = str(free_port())
    result_queue: mp.Queue = mp.Queue()

    def worker(local_rank: int) -> None:
        ensure_runtime_dirs(runtime_dir)
        rank = local_rank

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        import torch.distributed as dist
        from vllm import LLM, SamplingParams

        dist.init_process_group(
            backend="cpu:gloo,npu:hccl",
            world_size=world_size,
            rank=rank,
        )

        llm = LLM(
            model=model,
            tensor_parallel_size=settings["tp_size"],
            distributed_executor_backend="external_launcher",
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            compilation_config=build_compile_dict(
                mode_token, runtime_dir / "vllm-cache" / f"{layout}_{mode_token}"
            ),
            **{
                "_".join(["enable", "expert", "parallel"]): settings["ep_enabled"],
            },
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=False,
        )

        prompt_char_counts = [len(prompt) for prompt in prompts]
        for _ in range(warmup_iters):
            dist.barrier()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            dist.barrier()

        latencies = []
        output_tokens_per_iter = []
        for _ in range(timed_iters):
            dist.barrier()
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - started
            dist.barrier()
            total_output_tokens = sum(
                len(output.outputs[0].token_ids) if output.outputs else 0 for output in outputs
            )
            if rank == 0:
                latencies.append(elapsed)
                output_tokens_per_iter.append(total_output_tokens)

        if rank == 0:
            result_queue.put(
                {
                    "latencies": latencies,
                    "total_output_tokens": output_tokens_per_iter,
                    "num_prompts": len(prompts),
                    "prompt_char_count": sum(prompt_char_counts),
                }
            )

        dist.destroy_process_group()

    procs = [mp.Process(target=worker, args=(rank,)) for rank in range(world_size)]
    for proc in procs:
        proc.start()

    payload = result_queue.get(timeout=7200)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=7200)
        if proc.exitcode:
            exit_code = proc.exitcode
    if exit_code:
        raise RuntimeError(f"{layout} failed with exit code {exit_code}")

    total_output_tokens = payload["total_output_tokens"][0] if payload["total_output_tokens"] else 0
    mean_latency = statistics.mean(payload["latencies"])
    return {
        "layout": layout,
        "engine_mode": mode_token,
        "num_prompts": payload["num_prompts"],
        "prompt_char_count": payload["prompt_char_count"],
        "timed_iters": timed_iters,
        "mean_latency_sec": mean_latency,
        "median_latency_sec": statistics.median(payload["latencies"]),
        "p95_latency_sec": percentile(payload["latencies"], 95),
        "total_output_tokens_per_iter": total_output_tokens,
        "output_tokens_per_sec": total_output_tokens / mean_latency if mean_latency else 0.0,
    }


def run_layout_c(
    *,
    model: str,
    prompts: list[str],
    mode_token: str,
    runtime_dir: Path,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_tokens: int,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    dp_size = 2
    master_ip = "127.0.0.1"
    master_port = str(free_port())
    result_queue: mp.Queue = mp.Queue()

    def worker(local_dp_rank: int) -> None:
        ensure_runtime_dirs(runtime_dir / f"dp_rank_{local_dp_rank}")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(local_dp_rank)
        os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = master_port

        from vllm import LLM, SamplingParams

        local_prompts = split_evenly(prompts, dp_size, local_dp_rank)
        if not local_prompts:
            local_prompts = ["Placeholder prompt."]

        llm = LLM(
            model=model,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            compilation_config=build_compile_dict(
                mode_token, runtime_dir / "vllm-cache" / f"layout_c_{mode_token}"
            ),
            **{
                "_".join(["enable", "expert", "parallel"]): True,
            },
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=False,
        )

        for _ in range(warmup_iters):
            llm.generate(local_prompts, sampling_params, use_tqdm=False)

        latencies = []
        output_tokens_per_iter = []
        for _ in range(timed_iters):
            started = time.perf_counter()
            outputs = llm.generate(local_prompts, sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - started
            total_output_tokens = sum(
                len(output.outputs[0].token_ids) if output.outputs else 0 for output in outputs
            )
            latencies.append(elapsed)
            output_tokens_per_iter.append(total_output_tokens)

        result_queue.put(
            {
                "rank": local_dp_rank,
                "num_prompts": len(local_prompts),
                "prompt_char_count": sum(len(prompt) for prompt in local_prompts),
                "latencies": latencies,
                "total_output_tokens": output_tokens_per_iter,
            }
        )

    procs = [mp.Process(target=worker, args=(rank,)) for rank in range(dp_size)]
    for proc in procs:
        proc.start()

    payloads = [result_queue.get(timeout=7200) for _ in range(dp_size)]

    exit_code = 0
    for proc in procs:
        proc.join(timeout=7200)
        if proc.exitcode:
            exit_code = proc.exitcode
    if exit_code:
        raise RuntimeError(f"layout_c failed with exit code {exit_code}")

    per_iter_latencies = []
    per_iter_output_tokens = []
    for i in range(timed_iters):
        iter_elapsed = max(payload["latencies"][i] for payload in payloads)
        iter_output_tokens = sum(payload["total_output_tokens"][i] for payload in payloads)
        per_iter_latencies.append(iter_elapsed)
        per_iter_output_tokens.append(iter_output_tokens)

    mean_latency = statistics.mean(per_iter_latencies)
    mean_output_tokens = statistics.mean(per_iter_output_tokens)
    return {
        "layout": "layout_c",
        "engine_mode": mode_token,
        "num_prompts": sum(payload["num_prompts"] for payload in payloads),
        "prompt_char_count": sum(payload["prompt_char_count"] for payload in payloads),
        "timed_iters": timed_iters,
        "mean_latency_sec": mean_latency,
        "median_latency_sec": statistics.median(per_iter_latencies),
        "p95_latency_sec": percentile(per_iter_latencies, 95),
        "total_output_tokens_per_iter": mean_output_tokens,
        "output_tokens_per_sec": mean_output_tokens / mean_latency if mean_latency else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/cache/hf_model/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[1] / "data" / "q3_ascend_decode_prompts.txt"),
    )
    parser.add_argument("--layout", default="layout_b", choices=["layout_a", "layout_b", "layout_c"])
    parser.add_argument("--engine-mode", default="mode_a", choices=["mode_a", "mode_b"])
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=1)
    parser.add_argument("--runtime-root", default="/home/ma-user/work/q3_ascend_student_runtime")
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "q3_baseline_result.json"),
    )

    args = parser.parse_args()

    prompts = read_prompts(Path(args.dataset))[: args.num_prompts]
    if not prompts:
        raise ValueError("Dataset is empty.")

    runtime_dir = Path(args.runtime_root)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if args.layout in {"layout_a", "layout_b"}:
        result = run_layout_ab(
            model=args.model,
            prompts=prompts,
            layout=args.layout,
            mode_token=args.engine_mode,
            runtime_dir=runtime_dir,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
        )
    else:
        result = run_layout_c(
            model=args.model,
            prompts=prompts,
            mode_token=args.engine_mode,
            runtime_dir=runtime_dir,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
