"""Validate that DepthCache preserves predictions on a complete data split."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import ImageFolder
from .evaluate import load_trained


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--schedule", default="1,.5,.25,.125")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, ckpt = load_trained(args.checkpoint, device)
    schedule = tuple(float(x) for x in args.schedule.split(","))
    dataset = ImageFolder(
        os.path.join(args.data, "val"),
        train=False,
        class_to_idx=ckpt.get("class_to_idx"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    count = agreement = ref_correct = cache_correct = halt_equal = 0
    max_abs = 0.0
    sum_abs = 0.0
    logit_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reference = model(
                images,
                steps=len(schedule),
                halting=True,
                impl="gather",
                capacity_schedule=schedule,
            )
            cached = model(
                images,
                steps=len(schedule),
                halting=True,
                impl="gather",
                capacity_schedule=schedule,
                cache_kv=True,
            )
        diff = (reference.logits.float() - cached.logits.float()).abs()
        ref_pred = reference.logits.argmax(dim=-1)
        cache_pred = cached.logits.argmax(dim=-1)
        batch = labels.numel()
        count += batch
        agreement += int((ref_pred == cache_pred).sum())
        ref_correct += int((ref_pred == labels).sum())
        cache_correct += int((cache_pred == labels).sum())
        halt_equal += int((reference.halt_step == cached.halt_step).all(dim=1).sum())
        max_abs = max(max_abs, float(diff.max()))
        sum_abs += float(diff.sum())
        logit_count += diff.numel()

    result = {
        "checkpoint": args.checkpoint,
        "schedule": schedule,
        "images": count,
        "prediction_agreement": agreement / count,
        "reference_accuracy": ref_correct / count,
        "cached_accuracy": cache_correct / count,
        "halt_map_agreement": halt_equal / count,
        "max_abs_logit_difference": max_abs,
        "mean_abs_logit_difference": sum_abs / logit_count,
        "precision": "bfloat16 autocast",
    }
    print(json.dumps(result), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
