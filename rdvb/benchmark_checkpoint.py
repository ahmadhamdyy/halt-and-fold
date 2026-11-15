"""Model-only latency for a trained checkpoint on real preloaded images.

Unlike the dataset evaluator, this excludes image decoding and host-to-device
copies.  It reports both the automatic execution policy and forced query
gathering so wall-clock claims cannot be confused with data-loader throughput.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .data import ImageFolder
from .evaluate import load_trained


@torch.inference_mode()
def time_model(model, images, steps, halting, impl, warmup, iters, capacity_schedule=None):
    def call():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                images,
                steps=steps,
                halting=halting,
                impl=impl,
                capacity_schedule=capacity_schedule,
            )

    for _ in range(warmup):
        out = call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = call()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return {
        "batch_ms": ms,
        "ms_per_image": ms / images.shape[0],
        "images_per_s": 1000.0 * images.shape[0] / ms,
        "ponder": float(out.ponder_hard.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, ckpt = load_trained(args.checkpoint, device)
    class_to_idx = ckpt.get("class_to_idx")
    cases = ((224, 64), (448, 8), (672, 2))
    rows = []
    for size, batch in cases:
        dataset = ImageFolder(os.path.join(args.data, "val"), train=False, class_to_idx=class_to_idx, image_size=size)
        images = torch.stack([dataset[i][0] for i in range(batch)]).to(device)
        dense = time_model(model, images, args.steps, False, "auto", args.warmup, args.iters)
        for impl in ("auto", "gather"):
            routed = time_model(model, images, args.steps, True, impl, args.warmup, args.iters)
            row = {
                "resolution": size,
                "batch": batch,
                "tokens": 1 + (size // model.cfg.patch_size) ** 2,
                "steps": args.steps,
                "impl": impl,
                "dense": dense,
                "routed": routed,
                "speedup": dense["batch_ms"] / routed["batch_ms"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        schedule = (1.0, 0.75, 0.5, 0.25)
        budgeted = time_model(
            model,
            images,
            args.steps,
            True,
            "gather",
            args.warmup,
            args.iters,
            capacity_schedule=schedule,
        )
        row = {
            "resolution": size,
            "batch": batch,
            "tokens": 1 + (size // model.cfg.patch_size) ** 2,
            "steps": args.steps,
            "impl": "budgeted_gather",
            "capacity_schedule": schedule,
            "dense": dense,
            "routed": budgeted,
            "speedup": dense["batch_ms"] / budgeted["batch_ms"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"gpu": torch.cuda.get_device_name(0), "checkpoint": args.checkpoint, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
