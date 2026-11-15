"""Staged training for the recurrent backbone.

Stage 1 trains a dense, depth-conditioned recurrent model so that the shared
core cannot evade learning by halting immediately.  Stage 2 freezes the visual
backbone and learns the router by matching the dense model while annealing a
compute target.  A short joint stage then adapts the model to its routed state.

Training can resume at phase boundaries and records metrics for each phase.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageFolder
from .model import RDVB, RDVBConfig, count_parameters
from .train import evaluate, lr_at, param_groups, set_seed


def make_model(num_classes: int) -> RDVB:
    cfg = RDVBConfig(
        num_classes=num_classes,
        embed_dim=512,
        num_heads=8,
        depth_entry=4,
        depth_exit=3,
        router="spatial",
        router_bias=-4.0,
        core_mode="film",
        cond_dim=128,
    )
    return RDVB(cfg)


def mixed_ce(logits, a, b, lam: float, smoothing: float = 0.1):
    first = F.cross_entropy(logits, a, label_smoothing=smoothing)
    if lam >= 1.0:
        return first
    second = F.cross_entropy(logits, b, label_smoothing=smoothing)
    return lam * first + (1.0 - lam) * second


def mix_batch(images, labels, alpha: float):
    if alpha <= 0:
        return images, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    order = torch.randperm(images.shape[0], device=images.device)
    return lam * images + (1.0 - lam) * images[order], labels[order], lam


def depth_loss(out, labels, labels_b, lam: float, aux_weight: float):
    final = mixed_ce(out.logits, labels, labels_b, lam)
    aux = final.new_zeros(())
    if out.aux_logits and aux_weight > 0:
        # Later recurrent states receive more weight.  A small auxiliary term
        # makes shallow budgets usable without forcing every depth to be equal.
        weights = torch.arange(1, len(out.aux_logits) + 1, device=final.device, dtype=final.dtype)
        losses = torch.stack([mixed_ce(x, labels, labels_b, lam) for x in out.aux_logits])
        aux = (losses * weights).sum() / weights.sum()
    return final + aux_weight * aux, final, aux


def distill_loss(student, teacher, target_compute: float, compute_weight: float):
    temp = 2.0
    kd = F.kl_div(
        F.log_softmax(student.logits / temp, dim=-1),
        F.softmax(teacher.logits / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
    # Match normalized patch features for transfer to dense heads.
    s = F.normalize(student.tokens[:, 1:].float(), dim=-1)
    t = F.normalize(teacher.tokens[:, 1:].float(), dim=-1)
    feature = (s - t).square().mean()
    compute = (student.ponder_soft.mean() - target_compute).square()
    return kd + 10.0 * feature + compute_weight * compute, kd, feature, compute


@torch.no_grad()
def curve(model, loader, device, thresholds=(0.35, 0.5, 0.65)):
    rows = []
    original = model.cfg.halt_threshold
    for steps in (1, 2, 4, 6, 8):
        dense = evaluate(model, loader, device, steps, False)
        rows.append({"steps": steps, "halting": False, **dense})
        if steps > 1:
            for threshold in thresholds:
                model.cfg.halt_threshold = threshold
                routed = evaluate(model, loader, device, steps, True)
                rows.append({"steps": steps, "halting": True, "threshold": threshold, **routed})
    model.cfg.halt_threshold = original
    return rows


def save_checkpoint(path, model, optimizer, epoch, phase, best, args, class_to_idx):
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "best": best,
            "args": vars(args),
            "class_to_idx": class_to_idx,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dense-epochs", type=int, default=80)
    parser.add_argument("--router-epochs", type=int, default=16)
    parser.add_argument("--joint-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--joint-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--target-compute", type=float, default=2.5)
    parser.add_argument("--compute-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
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
    model = make_model(len(train_set.class_to_idx)).to(device)
    start_epoch = 0
    best = -1.0
    best_routed = -1.0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        start_epoch = int(resume["epoch"]) + 1
        best = float(resume.get("best", -1.0))

    dense_end = args.dense_epochs
    router_end = dense_end + args.router_epochs
    total_epochs = router_end + args.joint_epochs
    log_path = os.path.join(args.out, "train.jsonl")
    meta = {
        "params": count_parameters(model),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "gpu": torch.cuda.get_device_name(0),
        "cfg": model.cfg.to_dict(),
        "args": vars(args),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta), flush=True)

    optimizer = None
    current_phase = None
    global_in_phase = 0
    for epoch in range(start_epoch, total_epochs):
        if epoch < dense_end:
            phase = "dense"
            phase_epoch = epoch
            phase_epochs = args.dense_epochs
            base_lr = args.lr
        elif epoch < router_end:
            phase = "router"
            phase_epoch = epoch - dense_end
            phase_epochs = args.router_epochs
            base_lr = args.router_lr
        else:
            phase = "joint"
            phase_epoch = epoch - router_end
            phase_epochs = args.joint_epochs
            base_lr = args.joint_lr

        if phase != current_phase:
            if phase == "router":
                # Begin routing from a neutral probability rather than the
                # conservative initialization used during dense training.
                model.router.reset_parameters(0.0)
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in model.router.parameters():
                    p.requires_grad_(True)
            elif phase == "joint":
                for p in model.parameters():
                    p.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                param_groups(model, args.weight_decay),
                lr=base_lr,
                fused=True,
            )
            current_phase = phase
            global_in_phase = 0

        model.train()
        losses = []
        ponders = []
        t0 = time.time()
        total_phase_steps = max(1, phase_epochs * len(train_loader))
        warmup = len(train_loader) if phase == "dense" else 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, labels_b, lam = mix_batch(images, labels, args.mixup if phase != "router" else 0.0)
            lr = lr_at(global_in_phase, total_phase_steps, warmup, base_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if phase == "dense":
                    out = model(images, steps=4, halting=False)
                    loss, ce, aux = depth_loss(out, labels, labels_b, lam, args.aux_weight)
                else:
                    with torch.no_grad():
                        teacher = model(images, steps=4, halting=False)
                    out = model(images, steps=4, halting=True)
                    ce = mixed_ce(out.logits, labels, labels_b, lam, smoothing=0.1)
                    if phase == "router":
                        alpha = min(1.0, (phase_epoch + 1) / max(1, phase_epochs // 2))
                        target = 4.0 + alpha * (args.target_compute - 4.0)
                    else:
                        # Joint adaptation must preserve the budget learned by
                        # the router phase; re-annealing here makes compute
                        # jump upward at the phase boundary.
                        target = args.target_compute
                    distill, kd, feature, compute = distill_loss(
                        out, teacher, target, args.compute_weight
                    )
                    loss = ce + distill
                    aux = kd
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss in {phase} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            ponders.append(float(out.ponder_soft.mean().detach()))
            global_in_phase += 1

        halting = phase != "dense"
        val = evaluate(model, val_loader, device, 4, halting)
        dense_val = evaluate(model, val_loader, device, 4, False)
        record = {
            "epoch": epoch,
            "phase": phase,
            "phase_epoch": phase_epoch,
            "seconds": time.time() - t0,
            "loss": sum(losses) / len(losses),
            "train_ponder_soft": sum(ponders) / len(ponders),
            "val_acc": val["acc"],
            "val_ponder": val["ponder_hard"],
            "dense_val_acc": dense_val["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        save_checkpoint(
            os.path.join(args.out, "last.pt"),
            model,
            optimizer,
            epoch,
            phase,
            best,
            args,
            train_set.class_to_idx,
        )
        if val["acc"] > best:
            best = val["acc"]
            save_checkpoint(
                os.path.join(args.out, "best.pt"),
                model,
                optimizer,
                epoch,
                phase,
                best,
                args,
                train_set.class_to_idx,
            )
        if phase != "dense" and val["acc"] > best_routed:
            best_routed = val["acc"]
            save_checkpoint(
                os.path.join(args.out, "best_routed.pt"),
                model,
                optimizer,
                epoch,
                phase,
                best_routed,
                args,
                train_set.class_to_idx,
            )

    routed_path = os.path.join(args.out, "best_routed.pt")
    final_path = routed_path if os.path.exists(routed_path) else os.path.join(args.out, "best.pt")
    best_ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    model.to(device).eval()
    result = {
        "best_any": best,
        "best_routed": best_routed,
        "selected_checkpoint": final_path,
        "best_epoch": best_ckpt["epoch"],
        "curve": curve(model, val_loader, device),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
