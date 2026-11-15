"""Wall-clock comparison of dense recurrence and query-sparse recurrence.

Classification resolution (224) has only 197 tokens, so attention is a small
part of the step. The same block at dense-prediction resolutions is where
gather/scatter into FlashAttention changes measured latency.
"""

from __future__ import annotations

import argparse
import json

import torch

from .model import RDVB, RDVBConfig


def time_call(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("ti", "s"), default="s")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    from .model import PRESETS

    preset = PRESETS[args.preset]
    cfg = RDVBConfig(embed_dim=preset["embed_dim"], num_heads=preset["num_heads"], num_classes=10, router="spatial")
    model = RDVB(cfg).to(device).eval()
    cases = [
        ("224", 8, 224),
        ("448", 4, 448),
        ("672", 2, 672),
        ("1024", 1, 1024),
    ]
    fractions = [1.0, 0.5, 0.25]
    rows = []
    sdp = {
        "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math": bool(torch.backends.cuda.math_sdp_enabled()),
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps({"sdp": sdp}), flush=True)
    for name, batch, size in cases:
        images = torch.randn(batch, 3, size, size, device=device)
        tokens = 1 + (size // 16) * (size // 16)

        def dense():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(images, steps=args.steps, halting=False)

        dense_ms = time_call(dense, args.warmup, args.iters)
        for frac in fractions:
            mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
            mask[:, 0] = True
            keep = max(1, int(round(frac * (tokens - 1))))
            mask[:, 1 : 1 + keep] = True

            def sparse(mask=mask):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(images, steps=args.steps, halting=False, fixed_active=mask, impl="gather")

            sparse_ms = time_call(sparse, args.warmup, args.iters)
            row = {
                "resolution": name,
                "batch": batch,
                "tokens": tokens,
                "steps": args.steps,
                "active_frac": frac,
                "dense_ms": dense_ms,
                "sparse_ms": sparse_ms,
                "speedup": dense_ms / sparse_ms,
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
        del images
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump({"sdp": sdp, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
