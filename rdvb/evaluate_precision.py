"""FP32/BF16 numerical agreement for full and folded execution."""

from __future__ import annotations

import argparse
import contextlib
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import ImageFolder
from .evaluate import load_trained


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--schedule", default="1,.5,.25,.125,.0625,.03125,.015625,.0078125")
    args = parser.parse_args()
    model, checkpoint = load_trained(args.checkpoint, torch.device("cuda"))
    dataset = ImageFolder(
        os.path.join(args.data, "val"), False, class_to_idx=checkpoint.get("class_to_idx")
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    schedule = tuple(float(value) for value in args.schedule.split(","))
    stats = {
        name: {"correct": 0, "agreement": 0, "halt_agreement": 0, "sum_diff": 0.0,
               "max_diff": 0.0, "values": 0}
        for name in ("fp32", "bf16")
    }
    cross_agreement = cross_halt_agreement = total = 0
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = {}
        common = dict(steps=len(schedule), halting=True, impl="gather", capacity_schedule=schedule)
        for name, context in (
            ("fp32", contextlib.nullcontext()),
            ("bf16", torch.autocast(device_type="cuda", dtype=torch.bfloat16)),
        ):
            with context:
                full = model(images, **common)
                folded = model(images, fold_halted=True, **common)
            full_pred = full.logits.argmax(-1)
            folded_pred = folded.logits.argmax(-1)
            diff = (full.logits.float() - folded.logits.float()).abs()
            row = stats[name]
            row["correct"] += int((folded_pred == labels).sum())
            row["agreement"] += int((full_pred == folded_pred).sum())
            row["halt_agreement"] += int((full.halt_step == folded.halt_step).all(1).sum())
            row["sum_diff"] += float(diff.sum())
            row["max_diff"] = max(row["max_diff"], float(diff.max()))
            row["values"] += diff.numel()
            outputs[name] = (folded_pred, folded.halt_step)
        total += labels.numel()
        cross_agreement += int((outputs["fp32"][0] == outputs["bf16"][0]).sum())
        cross_halt_agreement += int((outputs["fp32"][1] == outputs["bf16"][1]).all(1).sum())
    result = {"checkpoint": args.checkpoint, "schedule": schedule, "images": total}
    for name, row in stats.items():
        result[name] = {
            "folded_accuracy": row["correct"] / total,
            "full_fold_prediction_agreement": row["agreement"] / total,
            "full_fold_halt_map_agreement": row["halt_agreement"] / total,
            "mean_abs_logit_difference": row["sum_diff"] / row["values"],
            "max_abs_logit_difference": row["max_diff"],
        }
    result["fp32_bf16_folded_prediction_agreement"] = cross_agreement / total
    result["fp32_bf16_folded_halt_map_agreement"] = cross_halt_agreement / total
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
