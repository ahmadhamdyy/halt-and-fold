"""Quality and numerical agreement for full and halt-and-fold execution."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import ImageFolder
from .evaluate import load_trained


@torch.inference_mode()
def evaluate(model, loader, schedule):
    total = full_correct = folded_correct = stale_correct = agreement = halt_agreement = 0
    patch_update_sum = 0.0
    max_diff = sum_diff = 0.0
    values = 0
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        common = dict(
            steps=len(schedule),
            halting=True,
            impl="gather",
            capacity_schedule=schedule,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            full = model(images, **common)
            folded = model(images, fold_halted=True, **common)
            stale = model(images, fold_halted=True, fold_writeback="stale", **common)
        full_pred = full.logits.argmax(-1)
        folded_pred = folded.logits.argmax(-1)
        stale_pred = stale.logits.argmax(-1)
        diff = (full.logits.float() - folded.logits.float()).abs()
        total += labels.numel()
        full_correct += int((full_pred == labels).sum())
        folded_correct += int((folded_pred == labels).sum())
        stale_correct += int((stale_pred == labels).sum())
        agreement += int((full_pred == folded_pred).sum())
        halt_agreement += int((full.halt_step == folded.halt_step).all(dim=1).sum())
        patch_update_sum += float(full.halt_step[:, 1:].float().mean(1).sum())
        max_diff = max(max_diff, float(diff.max()))
        sum_diff += float(diff.sum())
        values += diff.numel()
    return {
        "schedule": schedule,
        "steps": len(schedule),
        "images": total,
        "mean_patch_updates": patch_update_sum / total,
        "full_accuracy": full_correct / total,
        "folded_accuracy": folded_correct / total,
        "stale_writeback_accuracy": stale_correct / total,
        "prediction_agreement": agreement / total,
        "halt_map_agreement": halt_agreement / total,
        "mean_abs_logit_difference": sum_diff / values,
        "max_abs_logit_difference": max_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--schedules",
        default="1,.5,.25,.125;1,.5,.25,.125,.0625,.03125,.015625,.0078125",
        help="semicolon-separated capacity schedules",
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
    schedules = tuple(
        tuple(float(value) for value in item.split(","))
        for item in args.schedules.split(";")
    )
    rows = []
    for schedule in schedules:
        row = evaluate(model, loader, schedule)
        print(json.dumps(row), flush=True)
        rows.append(row)
    result = {"checkpoint": args.checkpoint, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
