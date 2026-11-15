"""Adapt a pretrained recurrent softmax core to halt-and-fold linear attention."""

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
from .finetune_capacity import parse_schedule


@torch.inference_mode()
def validate(model, loader, schedule, folded=False):
    model.eval()
    correct = total = 0
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
                fold_halted=folded,
            )
        correct += int((out.logits.argmax(-1) == labels).sum())
        total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--teacher-steps", type=int, default=0)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--schedule", default="1,.5,.25,.125")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    schedule = parse_schedule(args.schedule)
    device = torch.device("cuda")

    student, source = load_trained(args.checkpoint, device)
    teacher_path = args.teacher_checkpoint or args.checkpoint
    teacher, _ = load_trained(teacher_path, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.cfg.core_attention = "linear"
    student.core.block.attn.kind = "linear"

    train_set = ImageFolder(
        os.path.join(args.data, "train"), True, class_to_idx=source.get("class_to_idx")
    )
    val_set = ImageFolder(
        os.path.join(args.data, "val"), False, class_to_idx=source.get("class_to_idx")
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    baseline = validate(student, val_loader, schedule)
    baseline_folded = validate(student, val_loader, schedule, folded=True)
    print(
        json.dumps(
            {"epoch": -1, "val_acc": baseline, "folded_val_acc": baseline_folded}
        ),
        flush=True,
    )
    best = baseline

    for epoch in range(args.epochs):
        student.train()
        losses = []
        before = time.time()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher_steps = args.teacher_steps or len(schedule)
                teacher_out = teacher(images, steps=teacher_steps, halting=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_out = student(
                    images,
                    steps=len(schedule),
                    halting=True,
                    impl="gather",
                    capacity_schedule=schedule,
                )
                ce = F.cross_entropy(student_out.logits, labels, label_smoothing=0.1)
                temperature = 2.0
                kd = F.kl_div(
                    F.log_softmax(student_out.logits / temperature, dim=-1),
                    F.softmax(teacher_out.logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature**2
                feature = (
                    F.normalize(student_out.tokens[:, 1:].float(), dim=-1)
                    - F.normalize(teacher_out.tokens[:, 1:].float(), dim=-1)
                ).square().mean()
                loss = ce + kd + 10.0 * feature
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * (epoch + 1) / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        acc = validate(student, val_loader, schedule)
        row = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "val_acc": acc,
            "seconds": time.time() - before,
            "lr": lr,
        }
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": student.state_dict(),
            "cfg": student.cfg.to_dict(),
            "class_to_idx": source.get("class_to_idx"),
            "schedule": schedule,
            "epoch": epoch,
            "val_acc": acc,
            "source": args.checkpoint,
            "arch": "rdvb",
        }
        torch.save(checkpoint, os.path.join(args.out, "last.pt"))
        if acc >= best:
            best = acc
            torch.save(checkpoint, os.path.join(args.out, "best.pt"))

    best_model, _ = load_trained(os.path.join(args.out, "best.pt"), device)
    folded = validate(best_model, val_loader, schedule, folded=True)
    result = {
        "baseline": baseline,
        "baseline_folded": baseline_folded,
        "best": best,
        "best_folded": folded,
        "schedule": schedule,
        "source": args.checkpoint,
    }
    print(json.dumps(result), flush=True)
    with open(os.path.join(args.out, "result.json"), "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
