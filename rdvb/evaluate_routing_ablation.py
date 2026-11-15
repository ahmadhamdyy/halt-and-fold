"""Matched-budget learned-versus-random routing evaluation."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from torch.utils.data import DataLoader

from .data import ImageFolder
from .evaluate import load_trained


@torch.inference_mode()
def evaluate(model, loader, schedule, routing_mode, seed):
    torch.manual_seed(seed)
    correct = total = 0
    updates = 0.0
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                images,
                steps=len(schedule),
                halting=True,
                impl="gather",
                capacity_schedule=schedule,
                fold_halted=True,
                routing_mode=routing_mode,
            )
        correct += int((output.logits.argmax(-1) == labels).sum())
        total += labels.numel()
        updates += float(output.halt_step[:, 1:].float().mean(1).sum())
    return {"accuracy": correct / total, "mean_patch_updates": updates / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seeds", default="101,102,103")
    parser.add_argument(
        "--schedules",
        default="1,.875,.75,.625,.5,.375,.25,.125;1,.75,.5,.375,.25,.125,.0625,.03125;1,.5,.25,.125,.0625,.03125,.015625,.0078125;1,.375,.1875,.09375,.046875,.0234375,.01171875,.0078125",
    )
    args = parser.parse_args()
    model, checkpoint = load_trained(args.checkpoint, torch.device("cuda"))
    dataset = ImageFolder(
        os.path.join(args.data, "val"), False, class_to_idx=checkpoint.get("class_to_idx")
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    seeds = [int(value) for value in args.random_seeds.split(",")]
    schedules = [tuple(float(v) for v in item.split(",")) for item in args.schedules.split(";")]
    rows = []
    for schedule in schedules:
        learned = evaluate(model, loader, schedule, "learned", 0)
        random_rows = [evaluate(model, loader, schedule, "random", seed) for seed in seeds]
        random_acc = [row["accuracy"] for row in random_rows]
        row = {
            "schedule": schedule,
            "mean_patch_updates": learned["mean_patch_updates"],
            "learned_accuracy": learned["accuracy"],
            "random_accuracy_mean": statistics.mean(random_acc),
            "random_accuracy_std": statistics.stdev(random_acc),
            "random_seeds": seeds,
            "random_rows": random_rows,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    result = {"checkpoint": args.checkpoint, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
