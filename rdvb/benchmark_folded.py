"""Compiled full-versus-folded normalized linear attention benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from .benchmark_compiled import measure
from .evaluate import load_trained


class LinearForward(torch.nn.Module):
    def __init__(self, model, schedule, folded: bool):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.folded = folded

    def forward(self, images):
        return self.model(
            images,
            steps=len(self.schedule),
            halting=True,
            impl="gather",
            capacity_schedule=self.schedule,
            fold_halted=self.folded,
        ).logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", default="512,1024,1344")
    parser.add_argument("--schedule", default="1,.5,.25,.125")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    model, _ = load_trained(args.checkpoint, torch.device("cuda"))
    model.cfg.core_attention = "linear"
    model.core.block.attn.kind = "linear"
    schedule = tuple(float(x) for x in args.schedule.split(","))
    rows = []
    for size in (int(x) for x in args.sizes.split(",")):
        images = torch.randn(1, 3, size, size, device="cuda")
        compiled = {}
        compile_seconds = {}
        for name, folded in (("full_linear", False), ("halt_and_fold", True)):
            wrapped = LinearForward(model, schedule, folded).eval()
            fn = torch.compile(wrapped, mode="default", fullgraph=False)
            before = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fn(images)
            torch.cuda.synchronize()
            compile_seconds[name] = time.time() - before
            compiled[name] = fn
        full_trials = [
            measure(compiled["full_linear"], images, args.warmup, args.iters)
            for _ in range(args.trials)
        ]
        folded_trials = [
            measure(compiled["halt_and_fold"], images, args.warmup, args.iters)
            for _ in range(args.trials)
        ]
        full_ms = statistics.mean(full_trials)
        folded_ms = statistics.mean(folded_trials)
        memory = {}
        for name in ("full_linear", "halt_and_fold"):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                compiled[name](images)
            torch.cuda.synchronize()
            memory[name] = max(0, torch.cuda.max_memory_allocated() - baseline) / (1024 ** 2)
        row = {
            "resolution": size,
            "tokens": 1 + (size // model.cfg.patch_size) ** 2,
            "schedule": schedule,
            "full_linear_ms": full_ms,
            "halt_and_fold_ms": folded_ms,
            "speedup": full_ms / folded_ms,
            "full_linear_std_ms": statistics.stdev(full_trials) if len(full_trials) > 1 else 0.0,
            "halt_and_fold_std_ms": statistics.stdev(folded_trials) if len(folded_trials) > 1 else 0.0,
            "full_linear_trials_ms": full_trials,
            "halt_and_fold_trials_ms": folded_trials,
            "full_linear_images_per_second": 1000.0 / full_ms,
            "halt_and_fold_images_per_second": 1000.0 / folded_ms,
            "incremental_peak_memory_mib": memory,
            "compile_seconds": compile_seconds,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint_weights": args.checkpoint,
        "note": "core attention switched to normalized ELU+1 linear attention",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
