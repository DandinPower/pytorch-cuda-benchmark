import argparse
import csv
from statistics import mean
from typing import List, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm.auto import tqdm

def parse_int_list(s: str) -> List[int]:
    """Parse comma‑separated integers (e.g. "1,4,8")."""
    if not s:
        raise argparse.ArgumentTypeError("List cannot be empty")
    try:
        return [int(token) for token in s.split(",") if token]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("All values must be integers") from exc


def get_dummy_inputs(batch_size: int, seq_len: int) -> torch.Tensor:
    """Allocate an input tensor of zeros on the current CUDA device."""
    return torch.zeros((batch_size, seq_len), dtype=torch.long, device="cuda")


def build_model(model_name: str, use_liger: bool) -> nn.Module:
    """Instantiate the specified model on CUDA, optionally with Liger kernel."""
    if use_liger:
        model_cls = AutoLigerKernelForCausalLM
    else:
        model_cls = AutoModelForCausalLM
    model: nn.Module = model_cls.from_pretrained(model_name, use_cache=False, device_map="cuda")
    model.eval()
    return model


def _time_forward(model: nn.Module, inputs: torch.Tensor) -> float:
    """Return latency (ms) of a single forward pass."""
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    _ = model(inputs)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt)


def write_csv(path: str, model_name: str, rows: List[dict]) -> None:
    """Write latency statistics to *path* in CSV format."""
    fieldnames = [
        "model_name",
        "batch_size",
        "context_length",
        "min_ms",
        "max_ms",
        "mean_ms",
    ]
    with open(path, "w", newline="", encoding="utf‑8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"model_name": model_name, **row})
            

def benchmark(
    model: nn.Module,
    batch_sizes: List[int],
    context_lengths: List[int],
    warmup_iters: int,
    test_iters: int,
) -> List[dict]:
    """Run latency measurements and return a list of result dictionaries."""

    torch.cuda.empty_cache()
    results: List[dict] = []

    combos: List[Tuple[int, int]] = [
        (bs, sl) for bs in batch_sizes for sl in context_lengths
    ]

    with torch.inference_mode():
        for bs, seq_len in tqdm(combos, desc="Configurations", unit="cfg"):
            inputs = get_dummy_inputs(bs, seq_len)

            for _ in range(warmup_iters):
                _ = model(inputs)
            torch.cuda.synchronize()

            latencies_ms: List[float] = []
            for _ in tqdm(
                range(test_iters),
                desc=f"bs={bs},len={seq_len}",
                unit="iter",
                leave=False,
            ):
                latencies_ms.append(_time_forward(model, inputs))

            results.append(
                {
                    "batch_size": bs,
                    "context_length": seq_len,
                    "min_ms": min(latencies_ms),
                    "max_ms": max(latencies_ms),
                    "mean_ms": mean(latencies_ms),
                }
            )

            del inputs
            torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--batch_sizes", type=parse_int_list, required=True,)
    parser.add_argument("--context_lengths", type=parse_int_list, required=True,)
    parser.add_argument("--warmup_iters", type=int, required=True,)
    parser.add_argument("--test_iters", type=int, required=True,)
    parser.add_argument("--output_csv", default="latency_results.csv")
    parser.add_argument("--liger_kernel", action="store_true", help="Enable Liger kernel")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print(f"Loading model {args.model!r} (liger_kernel={args.liger_kernel}) …", flush=True)
    model = build_model(args.model, args.liger_kernel)
    print("Model ready — starting benchmark\n", flush=True)

    rows = benchmark(
        model=model,
        batch_sizes=args.batch_sizes,
        context_lengths=args.context_lengths,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters,
    )

    write_csv(args.output_csv, args.model, rows)
    print(f"Done! Results saved to {args.output_csv}\n")


if __name__ == "__main__":
    main()
