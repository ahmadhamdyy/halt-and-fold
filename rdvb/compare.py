"""Train every baseline and the recurrent model on the same ImageNette recipe.

The recipe is fixed: 16 epochs, batch 96, AdamW 3e-4, cosine, label smoothing
0.1, seed 0, 224 crops. Method-specific losses stay on, because they are part
of the method. Reported latency is measured val-set wall clock on one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .baselines import build_baseline
from .data import ImageFolder
from .model import PRESETS, RDVB, RDVBConfig, count_parameters
from .train import batch_loss, evaluate, lr_at, param_groups, ponder_target, set_seed


METHODS = (
    "rdvb",
    "rdvb_shared",
    "plain3",
    "plain6",
    "plain12",
    "early_exit",
    "token_prune",
    "tome",
    "mod",
    "swin_t",
)


def build(name: str, num_classes: int, preset: str):
    if name == "rdvb":
        spec = PRESETS[preset]
        cfg = RDVBConfig(
            num_classes=num_classes,
            embed_dim=spec["embed_dim"],
            num_heads=spec["num_heads"],
            router="spatial",
            router_bias=-1.5,
        )
        return RDVB(cfg)
    if name == "rdvb_shared":
        spec = PRESETS[preset]
        cfg = RDVBConfig(
            num_classes=num_classes,
            embed_dim=spec["embed_dim"],
            num_heads=spec["num_heads"],
            router="none",
        )
        return RDVB(cfg)
    if name == "rdvb_v2":
        # Wider configuration with additional entry and exit blocks.
        cfg = RDVBConfig(
            num_classes=num_classes,
            embed_dim=512,
            num_heads=8,
            depth_entry=4,
            depth_exit=3,
            router="spatial",
            router_bias=-3.0,
        )
        return RDVB(cfg)
    if name == "rdvb_v3":
        # Dense eight-repeat variant of the v2 configuration.
        cfg = RDVBConfig(
            num_classes=num_classes,
            embed_dim=512,
            num_heads=8,
            depth_entry=4,
            depth_exit=3,
            router="spatial",
            router_bias=-3.0,
        )
        return RDVB(cfg)
    if name == "rdvb_v4":
        # Dense four-repeat variant trained from the final prediction.
        cfg = RDVBConfig(
            num_classes=num_classes,
            embed_dim=512,
            num_heads=8,
            depth_entry=4,
            depth_exit=3,
            router="spatial",
            router_bias=-3.0,
        )
        return RDVB(cfg)
    return build_baseline(name, num_classes, preset)


def uses_halting(name: str) -> bool:
    return name in ("rdvb", "rdvb_v2")


def train_halting(name: str) -> bool:
    return uses_halting(name)


def eval_steps(name: str) -> int:
    if name == "rdvb_v3":
        return 8
    return 4


def mixed_ce(logits, labels_a, labels_b, lam: float, smoothing: float):
    import torch.nn.functional as F

    loss_a = F.cross_entropy(logits, labels_a, label_smoothing=smoothing)
    if lam >= 1:
        return loss_a
    loss_b = F.cross_entropy(logits, labels_b, label_smoothing=smoothing)
    return lam * loss_a + (1.0 - lam) * loss_b


def loss_for(name: str, out, labels_a, labels_b, lam: float, epoch: int):
    if name == "rdvb":
        target = ponder_target(epoch, 4, 4, 2.0)
        return batch_loss(out, labels_a, 0.1, 1.0, 0.05, target)
    if name == "rdvb_v4":
        total = mixed_ce(out.logits, labels_a, labels_b, lam, 0.1)
        return total, {"loss": float(total.detach()), "ce": float(total.detach())}
    if name in ("rdvb_v2", "rdvb_v3"):
        ce = mixed_ce(out.logits, labels_a, labels_b, lam, 0.1)
        aux = out.logits.new_zeros(())
        if out.aux_logits:
            aux = sum(mixed_ce(logits, labels_a, labels_b, lam, 0.1) for logits in out.aux_logits)
            aux = aux / len(out.aux_logits)
        total = ce + 0.5 * aux
        return total, {"loss": float(total.detach()), "ce": float(ce.detach())}
    if name == "early_exit":
        losses = [mixed_ce(logits, labels_a, labels_b, lam, 0.1) for logits in out.aux_logits]
        total = sum(losses) / len(losses)
        stats = {"loss": float(total.detach()), "ce": float(total.detach())}
        return total, stats
    total = mixed_ce(out.logits, labels_a, labels_b, lam, 0.1)
    return total, {"loss": float(total.detach()), "ce": float(total.detach())}


def steps_for(name: str, epoch: int) -> int:
    if name == "rdvb_v3":
        return 8
    if name == "rdvb_v4":
        return 4
    if name not in ("rdvb", "rdvb_shared"):
        return 4
    # Same depth curriculum used for the recurrent model: 1, then up to 4.
    if epoch < 4:
        return max(1, epoch + 1)
    return 4


@torch.no_grad()
def timed_accuracy(model, loader, device, name: str) -> dict:
    model.eval()
    if name == "early_exit":
        model.exit_threshold = 0.7
    correct = 0
    total = 0
    ponder = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(images, steps=eval_steps(name), halting=uses_halting(name))
        correct += int((out.logits.argmax(-1) == labels).sum().item())
        total += labels.numel()
        ponder += float(out.ponder_hard.sum().item())
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000.0
    row = {
        "acc": correct / max(1, total),
        "compute_proxy": ponder / max(1, total),
        "ms_per_image": 1000.0 * seconds / max(1, total),
        "images_per_s": total / max(seconds, 1e-6),
    }
    if name == "early_exit":
        model.exit_threshold = 1.1
        full = evaluate(model, loader, device, eval_steps(name), False)
        row["acc_full"] = full["acc"]
        model.exit_threshold = 0.7
    if uses_halting(name):
        full = evaluate(model, loader, device, eval_steps(name), False)
        row["acc_full"] = full["acc"]
    return row


def train_one(name, loaders, device, args, num_classes: int) -> dict:
    set_seed(args.seed)
    model = build(name, num_classes, args.preset).to(device)
    train_loader, val_loader = loaders
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_epochs * steps_per_epoch)
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    best = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    global_step = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        if name == "early_exit":
            model.exit_threshold = 1.1
        t0 = time.time()
        running = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lr = lr_at(global_step, total_steps, warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            step_cap = steps_for(name, epoch)
            import random

            labels_b = labels
            lam = 1.0
            if args.mixup > 0:
                lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                index = torch.randperm(images.shape[0], device=images.device)
                images = lam * images + (1.0 - lam) * images[index]
                labels_b = labels[index]
            steps = random.randint(1, step_cap) if name == "rdvb" else step_cap
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(images, steps=steps, halting=train_halting(name))
                loss, stats = loss_for(name, out, labels, labels_b, lam, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} non-finite loss at epoch {epoch}: {stats}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(stats["loss"])
            global_step += 1
        val = evaluate(model, val_loader, device, steps_for(name, epoch), uses_halting(name))
        record = {
            "method": name,
            "epoch": epoch,
            "seconds": time.time() - t0,
            "train_loss": sum(running) / len(running),
            "val_acc": val["acc"],
            "val_ponder": val["ponder_hard"],
        }
        print(json.dumps(record), flush=True)
        history.append(record)
        ckpt = {"model": model.state_dict(), "method": name, "epoch": epoch, "val_acc": val["acc"]}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if val["acc"] > best:
            best = val["acc"]
            torch.save(ckpt, best_path)
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    model.to(device)
    measured = timed_accuracy(model, val_loader, device, name)
    summary = {
        "method": name,
        "params": count_parameters(model),
        "best_val_acc_during_train": best,
        **measured,
        "history": history,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--preset", default="s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0.0, help="Beta mixup alpha; 0 disables it")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    device = torch.device("cuda")
    train_set = ImageFolder(os.path.join(args.data, "train"), train=True)
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
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name in [item for item in args.methods.split(",") if item]:
        rows.append(train_one(name, (train_loader, val_loader), device, args, len(train_set.class_to_idx)))
    table = [{k: v for k, v in row.items() if k != "history"} for row in rows]
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps({"wrote": os.path.join(args.out, "comparison.json"), "n": len(table)}), flush=True)


if __name__ == "__main__":
    main()
