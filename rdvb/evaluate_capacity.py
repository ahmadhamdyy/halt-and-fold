"""Evaluate fixed-capacity compute dials on an ImageFolder validation set."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import ImageFolder
from .evaluate import load_trained


def parse_schedule(text):
    return tuple(float(x) for x in text.split(","))


@torch.inference_mode()
def evaluate(model, loader, schedule):
    correct = total = 0
    patch_ponder = 0.0
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                images,
                steps=len(schedule),
                halting=True,
                impl="gather",
                capacity_schedule=schedule,
            )
        correct += int((out.logits.argmax(-1) == labels).sum())
        total += labels.numel()
        patch_ponder += float(out.halt_step[:, 1:].float().mean(dim=1).sum())
    return {"schedule": schedule, "top1": correct / total, "patch_updates": patch_ponder / total, "n": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--schedules", default="1,.875,.75,.5;1,.75,.5,.25;1,.5,.25,.125")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, ckpt = load_trained(args.checkpoint, torch.device("cuda"))
    dataset = ImageFolder(
        os.path.join(args.data, "val"),
        train=False,
        class_to_idx=ckpt.get("class_to_idx"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    rows = [evaluate(model, loader, parse_schedule(x)) for x in args.schedules.split(";")]
    for row in rows:
        print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
