"""Train a recurrent-depth backbone, from scratch or from a converted ViT."""

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

from .data import ImageFolder, load_imagenet_synset_index
from .evaluate import load_trained
from .model import PRESETS, PlainViT, RDVB, RDVBConfig, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            nodecay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, total - warmup))
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def ponder_target(epoch: int, anneal_epochs: int, steps_max: int, final_target: float) -> float:
    if anneal_epochs <= 0:
        return final_target
    alpha = min(1.0, epoch / float(anneal_epochs))
    return steps_max * (1.0 - alpha) + final_target * alpha


def batch_loss(out, labels, smoothing: float, kd_weight: float, ponder_weight: float, target: float):
    ce = F.cross_entropy(out.logits, labels, label_smoothing=smoothing)
    aux = out.logits.new_zeros(())
    if out.aux_logits:
        aux = sum(F.cross_entropy(logits, labels, label_smoothing=smoothing) for logits in out.aux_logits)
        aux = aux / len(out.aux_logits)
    kd = out.logits.new_zeros(())
    if kd_weight > 0 and out.aux_logits:
        teacher = F.softmax(out.logits.detach(), dim=-1)
        kd = sum(
            F.kl_div(F.log_softmax(logits, dim=-1), teacher, reduction="batchmean") for logits in out.aux_logits
        )
        kd = kd / len(out.aux_logits)
    reg = out.logits.new_zeros(())
    if ponder_weight > 0:
        reg = (out.ponder_soft.mean() - target) ** 2
    total = ce + aux + kd_weight * kd + ponder_weight * reg
    stats = {
        "loss": float(total.detach()),
        "ce": float(ce.detach()),
        "aux": float(aux.detach()),
        "kd": float(kd.detach()),
        "ponder_reg": float(reg.detach()),
        "ponder_soft": float(out.ponder_soft.mean().detach()),
        "ponder_hard": float(out.ponder_hard.mean().detach()),
    }
    return total, stats


@torch.no_grad()
def evaluate(model, loader, device, steps: int, halting: bool) -> dict:
    model.eval()
    correct = 0
    total = 0
    ponder = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(images, steps=steps, halting=halting)
        pred = out.logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
        ponder += float(out.ponder_hard.sum().item())
    return {"acc": correct / max(1, total), "ponder_hard": ponder / max(1, total), "n": total}


def build_model(args, num_classes: int):
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
        cfg = RDVBConfig.from_dict(ckpt["cfg"])
        model = RDVB(cfg)
        model.load_state_dict(ckpt["model"])
        if cfg.num_classes != num_classes:
            model.head = torch.nn.Linear(cfg.embed_dim, num_classes)
            torch.nn.init.trunc_normal_(model.head.weight, std=0.02)
            torch.nn.init.zeros_(model.head.bias)
            model.cfg.num_classes = num_classes
        return model
    preset = PRESETS[args.preset]
    cfg = RDVBConfig(
        num_classes=num_classes,
        embed_dim=preset["embed_dim"],
        num_heads=preset["num_heads"],
        router=args.router,
        kv_mode=args.kv_mode,
        router_bias=args.router_bias,
        drop=args.drop,
    )
    if args.arch == "plain":
        return PlainViT(cfg, depth=args.plain_depth)
    return RDVB(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="directory with train/ and val/ ImageFolder splits")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arch", choices=("rdvb", "plain"), default="rdvb")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="s")
    parser.add_argument("--finetune", default="")
    parser.add_argument("--teacher", default="", help="frozen pretrained ViT used as a distillation teacher")
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--teacher-temp", type=float, default=2.0)
    parser.add_argument("--synset-index", default="", help="score ImageFolder synset names as ImageNet-1K indices")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--steps-min", type=int, default=1)
    parser.add_argument("--steps-max", type=int, default=4)
    parser.add_argument("--router", choices=("spatial", "independent", "none"), default="spatial")
    parser.add_argument("--kv-mode", choices=("frozen", "drop"), default="frozen")
    parser.add_argument("--router-bias", type=float, default=-1.5)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--plain-depth", type=int, default=6)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ponder-weight", type=float, default=0.05)
    parser.add_argument("--ponder-target", type=float, default=2.0)
    parser.add_argument("--ponder-anneal-epochs", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-depth-sampling", action="store_true")
    parser.add_argument(
        "--steps-curriculum",
        type=int,
        default=0,
        help="epochs spent raising the maximum recurrent depth from steps-min to steps-max",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=0, help="validation recurrence; default steps-max")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    class_to_idx = load_imagenet_synset_index(args.synset_index) if args.synset_index else None
    train_set = ImageFolder(os.path.join(args.data, "train"), train=True, class_to_idx=class_to_idx)
    val_set = ImageFolder(os.path.join(args.data, "val"), train=False, class_to_idx=train_set.class_to_idx)
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
    num_classes = len(train_set.class_to_idx) if class_to_idx is None else max(train_set.class_to_idx.values()) + 1
    model = build_model(args, num_classes=num_classes).to(device)
    teacher = None
    if args.teacher:
        teacher, _teacher_ckpt = load_trained(args.teacher, device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
    halting = isinstance(model, RDVB) and model.cfg.router != "none"
    if not halting:
        args.ponder_weight = 0.0
        args.distill_weight = args.distill_weight if isinstance(model, RDVB) else 0.0
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_epochs * steps_per_epoch)
    eval_steps = args.eval_steps or args.steps_max
    log_path = os.path.join(args.out, "train.log")
    meta = {
        "args": vars(args),
        "params": count_parameters(model),
        "classes": len(train_set.class_to_idx),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "halting": halting,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))

    best = -1.0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if args.steps_curriculum > 0:
            alpha = min(1.0, epoch / float(args.steps_curriculum))
            epoch_max_steps = int(round(args.steps_min + alpha * (args.steps_max - args.steps_min)))
            epoch_max_steps = max(args.steps_min, min(args.steps_max, epoch_max_steps))
        else:
            epoch_max_steps = args.steps_max
        target = ponder_target(epoch, args.ponder_anneal_epochs, epoch_max_steps, min(args.ponder_target, float(epoch_max_steps)))
        running = []
        t0 = time.time()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.no_depth_sampling or not isinstance(model, RDVB):
                steps = epoch_max_steps if isinstance(model, RDVB) else args.steps_max
            else:
                steps = random.randint(args.steps_min, epoch_max_steps)
            lr = lr_at(global_step, total_steps, warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(images, steps=steps, halting=halting)
                loss, stats = batch_loss(
                    out, labels, args.label_smoothing, args.distill_weight, args.ponder_weight, target
                )
                if teacher is not None:
                    with torch.no_grad():
                        target_logits = teacher(images).logits
                    temp = args.teacher_temp
                    distill = F.kl_div(
                        F.log_softmax(out.logits / temp, dim=-1),
                        F.softmax(target_logits / temp, dim=-1),
                        reduction="batchmean",
                    ) * (temp * temp)
                    loss = loss + args.teacher_weight * distill
                    stats["teacher_kd"] = float(distill.detach())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {global_step}: {stats}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            stats["lr"] = lr
            stats["steps"] = steps
            stats["target"] = target
            running.append(stats)
            global_step += 1
        val_steps = epoch_max_steps if isinstance(model, RDVB) else eval_steps
        val = evaluate(model, val_loader, device, val_steps, halting)
        means = {k: sum(row[k] for row in running) / len(running) for k in ("loss", "ce", "ponder_soft", "ponder_hard")}
        record = {
            "epoch": epoch,
            "seconds": time.time() - t0,
            "train_loss": means["loss"],
            "train_ce": means["ce"],
            "train_ponder_soft": means["ponder_soft"],
            "train_ponder_hard": means["ponder_hard"],
            "ponder_target": target,
            "val_acc": val["acc"],
            "val_ponder_hard": val["ponder_hard"],
            "eval_steps": val_steps,
            "epoch_max_steps": epoch_max_steps,
        }
        print(json.dumps(record), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        ckpt = {
            "model": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "arch": "plain" if isinstance(model, PlainViT) else "rdvb",
            "plain_depth": args.plain_depth,
            "class_to_idx": train_set.class_to_idx,
            "epoch": epoch,
            "val_acc": val["acc"],
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if val["acc"] > best:
            best = val["acc"]
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
    print(json.dumps({"best_val_acc": best, "out": args.out}))


if __name__ == "__main__":
    main()
