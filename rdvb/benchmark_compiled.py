"""Fair compiled-vs-compiled latency for fixed-capacity routing.

Fixed capacities make tensor shapes known to TorchInductor, allowing it to
fuse routing and packing operations.  This benchmark is intended for the
high-resolution regime where query-sparse attention can offset that overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .evaluate import load_trained


class FixedForward(torch.nn.Module):
    def __init__(self, model, schedule, cache_kv=False):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.cache_kv = cache_kv

    def forward(self, images):
        if self.schedule is None:
            return self.model(images, steps=4, halting=False).logits
        return self.model(
            images,
            steps=4,
            halting=True,
            impl="gather",
            capacity_schedule=self.schedule,
            cache_kv=self.cache_kv,
        ).logits


@torch.inference_mode()
def measure(fn, images, warmup, iters):
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fn(images)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fn(images)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def parse_schedule(text):
    values = tuple(float(x) for x in text.split(","))
    if len(values) != 4:
        raise ValueError("the compiled benchmark currently expects four capacities")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", default="1024,1344")
    parser.add_argument("--schedule", default="1,.5,.25,.125")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile-mode", default="default")
    args = parser.parse_args()
    model, _ = load_trained(args.checkpoint, torch.device("cuda"))
    schedule = parse_schedule(args.schedule)
    rows = []
    for size in (int(x) for x in args.sizes.split(",")):
        images = torch.randn(1, 3, size, size, device="cuda")
        compiled = {}
        compile_seconds = {}
        variants = (
            ("dense", None, False),
            ("budgeted", schedule, False),
            ("budgeted_cached", schedule, True),
        )
        for name, selected, cache_kv in variants:
            wrapped = FixedForward(model, selected, cache_kv).eval()
            fn = torch.compile(wrapped, mode=args.compile_mode, fullgraph=False)
            before = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fn(images)
            torch.cuda.synchronize()
            compile_seconds[name] = time.time() - before
            compiled[name] = fn
        dense_ms = measure(compiled["dense"], images, args.warmup, args.iters)
        budgeted_ms = measure(compiled["budgeted"], images, args.warmup, args.iters)
        cached_ms = measure(compiled["budgeted_cached"], images, args.warmup, args.iters)
        row = {
            "resolution": size,
            "batch": 1,
            "tokens": 1 + (size // model.cfg.patch_size) ** 2,
            "schedule": schedule,
            "mean_patch_updates": sum(schedule),
            "dense_ms": dense_ms,
            "budgeted_ms": budgeted_ms,
            "speedup": dense_ms / budgeted_ms,
            "budgeted_cached_ms": cached_ms,
            "cached_speedup": dense_ms / cached_ms,
            "cache_gain": budgeted_ms / cached_ms,
            "compile_seconds": compile_seconds,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint": args.checkpoint,
        "compile_mode": args.compile_mode,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

