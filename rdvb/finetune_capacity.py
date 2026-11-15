"""Fine-tune only the spatial router for a fixed deployment budget."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageFolder
from .evaluate import load_trained


def parse_schedule(text):
    return tuple(float(x) for x in text.split(","))


@torch.inference_mode()
def validate(model, loader, schedule):
    correct = total = 0
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(images, steps=len(schedule), halting=True, impl="gather", capacity_schedule=schedule)
        correct += int((out.logits.argmax(-1) == labels).sum())
        total += labels.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--schedule", default="1,.5,.25,.125")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    schedule = parse_schedule(args.schedule)
    model, source = load_trained(args.checkpoint, torch.device("cuda"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.router.parameters():
        parameter.requires_grad_(True)
    train_set = ImageFolder(os.path.join(args.data, "train"), True, class_to_idx=source.get("class_to_idx"))
    val_set = ImageFolder(os.path.join(args.data, "val"), False, class_to_idx=source.get("class_to_idx"))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=args.lr, weight_decay=0.01)
    best = validate(model, val_loader, schedule)
    print(json.dumps({"epoch": -1, "val_acc": best}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        before = time.time()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher = model(images, steps=len(schedule), halting=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student = model(images, steps=len(schedule), halting=True, impl="gather", capacity_schedule=schedule)
                ce = F.cross_entropy(student.logits, labels)
                temp = 2.0
                kd = F.kl_div(
                    F.log_softmax(student.logits / temp, dim=-1),
                    F.softmax(teacher.logits / temp, dim=-1),
                    reduction="batchmean",
                ) * (temp * temp)
                feature = (
                    F.normalize(student.tokens[:, 1:].float(), dim=-1)
                    - F.normalize(teacher.tokens[:, 1:].float(), dim=-1)
                ).square().mean()
                loss = ce + kd + 10.0 * feature
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        # Cosine decay over the short router-only run.
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * (epoch + 1) / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        acc = validate(model, val_loader, schedule)
        row = {"epoch": epoch, "loss": sum(losses) / len(losses), "val_acc": acc,
               "seconds": time.time() - before, "lr": lr}
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "class_to_idx": source.get("class_to_idx"),
            "schedule": schedule,
            "epoch": epoch,
            "val_acc": acc,
            "source": args.checkpoint,
        }
        torch.save(checkpoint, os.path.join(args.out, "last.pt"))
        if acc >= best:
            best = acc
            torch.save(checkpoint, os.path.join(args.out, "best.pt"))
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"best": best, "schedule": schedule, "source": args.checkpoint}, f, indent=2)


if __name__ == "__main__":
    main()
