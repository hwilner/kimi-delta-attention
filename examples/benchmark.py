"""Performance benchmark for the Kimi Delta Attention implementations.

Measures wall-clock latency and peak memory for the three KDA code paths
shipped in this repository against PyTorch's fused scaled-dot-product
attention (SDPA) as a reference point.

The numbers are produced by *this* script on the machine that runs it. They
are a smoke-test signal, not a published benchmark: the comparison is not
apples-to-apples (KDA is a recurrent linear-attention formulation while SDPA
is quadratic full attention), and no attempt is made to control for kernel
fusion, hardware, or thread count. Treat the output as "did it run and how
did it scale", never as a claim from the paper.

Usage:
    python examples/benchmark.py --seq-len 1024 --batch-size 4 --num-heads 8
"""

import argparse
import time

import torch

from kda import KimiDeltaAttention


def _synchronize(device: torch.device) -> None:
    """Wait for queued device work before timing or reading memory.

    CUDA kernels are launched asynchronously, so without this barrier the
    reported wall-clock time would exclude the work being measured.

    Args:
        device: Device whose queue should be drained.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward(layer: torch.nn.Module, x: torch.Tensor, device: torch.device,
                  warmup: int, iters: int) -> float:
    """Return the mean forward-pass latency in milliseconds.

    Args:
        layer: Module to time.
        x: Input activations of shape ``(B, L, D)``.
        device: Device the layer runs on.
        warmup: Number of untimed iterations to absorb lazy allocation and
            kernel selection before measurement begins.
        iters: Number of timed iterations to average over.

    Returns:
        Mean forward-pass latency in milliseconds.
    """
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(iters):
            layer(x)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / iters


def _time_sdpa(x: torch.Tensor, warmup: int, iters: int) -> float:
    """Return the mean latency of PyTorch SDPA over the same input.

    Args:
        x: Input activations of shape ``(B, L, D)``.
        warmup: Number of untimed iterations before measurement begins.
        iters: Number of timed iterations to average over.

    Returns:
        Mean forward-pass latency in milliseconds.
    """
    with torch.no_grad():
        for _ in range(warmup):
            torch.nn.functional.scaled_dot_product_attention(x, x, x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

        start = time.perf_counter()
        for _ in range(iters):
            torch.nn.functional.scaled_dot_product_attention(x, x, x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / iters


def _reset_peak_memory(device: torch.device) -> None:
    """Clear accumulated peak-memory statistics before a measurement.

    Args:
        device: Device whose memory statistics should be reset.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated device memory in mebibytes.

    Args:
        device: Device to query.

    Returns:
        Peak allocated memory in MiB, or 0.0 on non-CUDA devices where this
        statistic is not available.
    """
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def main() -> None:
    """Parse arguments, run the benchmark, and print a results table."""
    parser = argparse.ArgumentParser(
        description="Benchmark KDA implementations against PyTorch SDPA.",
    )
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length (default: 1024).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4).")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads (default: 8).")
    parser.add_argument("--d-model", type=int, default=None,
                        help="Model dimension (default: num-heads * 64).")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="Chunk length for the chunkwise paths (default: 64).")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Untimed warmup iterations (default: 2).")
    parser.add_argument("--iters", type=int, default=5,
                        help="Timed iterations averaged per implementation (default: 5).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for input generation (default: 0).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on, e.g. 'cpu' or 'cuda' (default: auto).")
    args = parser.parse_args()

    if args.d_model is None:
        args.d_model = args.num_heads * 64

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)

    # The state grows with d_model^2 per head, so keep the default bounded on
    # CPU where large allocations are slow.
    head_dim = args.d_model // args.num_heads
    print("=" * 72)
    print("Kimi Delta Attention benchmark")
    print("=" * 72)
    print(f"  device        : {device}")
    print(f"  batch size    : {args.batch_size}")
    print(f"  sequence len  : {args.seq_len}")
    print(f"  d_model       : {args.d_model}")
    print(f"  heads         : {args.num_heads} (head_dim={head_dim})")
    print(f"  chunk size    : {args.chunk_size}")
    print(f"  warmup/iters  : {args.warmup}/{args.iters}")
    print()

    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)

    variants = {
        "KDA (sequential)": dict(use_chunkwise=False),
        "KDA (chunkwise)": dict(use_chunkwise=True, chunk_size=args.chunk_size),
    }

    results = []
    for name, kwargs in variants.items():
        layer = KimiDeltaAttention(
            d_model=args.d_model, num_heads=args.num_heads,
            use_short_conv=False, **kwargs,
        ).to(device).eval()
        _reset_peak_memory(device)
        latency = _time_forward(layer, x, device, args.warmup, args.iters)
        results.append((name, latency, _peak_memory_mb(device)))

    # SDPA is measured in-place on the same activation to keep the input
    # identical across implementations.
    _reset_peak_memory(device)
    results.append((
        "SDPA (reference)",
        _time_sdpa(x, args.warmup, args.iters),
        _peak_memory_mb(device),
    ))

    print(f"{'implementation':<22} {'latency (ms)':>14} {'peak mem (MiB)':>16}")
    print("-" * 72)
    for name, latency, memory in results:
        mem_str = f"{memory:.1f}" if memory > 0 else "n/a"
        print(f"{name:<22} {latency:>14.3f} {mem_str:>16}")
    print("-" * 72)
    print(
        "Note: these are timings from this run on this machine. KDA is a\n"
        "recurrent linear-attention formulation and SDPA is quadratic full\n"
        "attention, so this table is a scale check, not a head-to-head claim."
    )


if __name__ == "__main__":
    main()
