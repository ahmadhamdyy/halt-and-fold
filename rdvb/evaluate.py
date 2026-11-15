"""Accuracy, latency, and halt-map measurements at several recurrence depths."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import IMAGENET_MEAN, IMAGENET_STD, ImageFolder, load_imagenet_synset_index
from .model import PlainViT, RDVB, RDVBConfig


def load_trained(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RDVBConfig.from_dict(ckpt["cfg"])
    if ckpt.get("arch") == "plain":
        model = PlainViT(cfg, depth=ckpt.get("plain_depth", 12))
    else:
        model = RDVB(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def sobel_mag(gray: torch.Tensor) -> torch.Tensor:
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def neighbor_variation(halt: torch.Tensor, gh: int, gw: int) -> float:
    grid = halt[:, 1:].float().reshape(-1, 1, gh, gw)
    dh = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).abs().mean()
    dw = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).abs().mean()
    return float(0.5 * (dh + dw))


@torch.no_grad()
def run_split(model, loader, device, steps, halting, adaptive, residual_tol, min_active):
    correct = 0
    total = 0
    ponder = 0.0
    patch_ponder = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                images,
                steps=steps,
                halting=halting,
                adaptive=adaptive,
                residual_tol=residual_tol,
                min_active=min_active,
            )
        pred = out.logits.float().argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
        ponder += float(out.ponder_hard.sum().item())
        patch_ponder += float(out.halt_step[:, 1:].float().sum().item())
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    tokens = out.halt_step.shape[1] - 1
    return {
        "steps": steps,
        "halting": halting,
        "adaptive": adaptive,
        "acc": correct / max(1, total),
        "n": total,
        "ponder_hard": ponder / max(1, total),
        "patch_ponder": patch_ponder / max(1, total * tokens),
        "seconds": elapsed,
        "images_per_s": total / max(elapsed, 1e-6),
        "ms_per_image": 1000.0 * elapsed / max(1, total),
    }


@torch.no_grad()
def analyze_halts(model, dataset: ImageFolder, device, steps: int, count: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    corrs = []
    variations = []
    shuffles = []
    saved = 0
    model.eval()
    for i in range(min(count, len(dataset))):
        image, _label = dataset[i]
        batch = image.unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch, steps=steps, halting=True)
        halt = out.halt_step
        gh = int(round((halt.shape[1] - 1) ** 0.5))
        gw = gh
        denorm = (image * std + mean).clamp(0, 1)
        gray = denorm.mean(dim=0, keepdim=True).unsqueeze(0)
        edges = sobel_mag(gray)
        pooled = torch.nn.functional.adaptive_avg_pool2d(edges, (gh, gw)).flatten()
        compute = halt[0, 1:].float().cpu()
        a = pooled.flatten().cpu() - pooled.mean().cpu()
        b = compute - compute.mean()
        corr = float((a * b).mean() / (a.std() * b.std() + 1e-8))
        corrs.append(corr)
        variations.append(neighbor_variation(halt.cpu(), gh, gw))
        perm = compute[torch.randperm(compute.numel())]
        grid = perm.reshape(1, 1, gh, gw)
        dh = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).abs().mean()
        dw = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).abs().mean()
        shuffles.append(float(0.5 * (dh + dw)))
        if saved < 8:
            heat = compute.reshape(gh, gw)
            heat_n = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            rgb = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            overlay = rgb.copy()
            heat_img = np.array(Image.fromarray((heat_n.numpy() * 255).astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
            overlay[..., 0] = np.clip(0.55 * overlay[..., 0] + 0.45 * heat_img, 0, 255).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(out_dir, f"halt_{i:02d}.png"))
            saved += 1
    return {
        "images": len(corrs),
        "edge_correlation_mean": float(np.mean(corrs)) if corrs else None,
        "neighbor_variation": float(np.mean(variations)) if variations else None,
        "shuffled_neighbor_variation": float(np.mean(shuffles)) if shuffles else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--steps", default="1,2,4,6,8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--halting", choices=("on", "off", "both"), default="both")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--residual-tol", type=float, default=0.05)
    parser.add_argument("--min-active", type=float, default=0.05)
    parser.add_argument("--label-mode", choices=("folder", "imagenet"), default="folder")
    parser.add_argument("--synset-index", default="")
    parser.add_argument("--class-to-idx", default="", help="json map used for folder labels; default sorted folders")
    parser.add_argument("--analyze", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, ckpt = load_trained(args.checkpoint, device)
    class_to_idx = None
    if args.label_mode == "folder":
        if args.class_to_idx:
            with open(args.class_to_idx) as f:
                class_to_idx = json.load(f)
        elif "class_to_idx" in ckpt:
            class_to_idx = ckpt["class_to_idx"]
    else:
        if not args.synset_index:
            raise SystemExit("--synset-index is required for imagenet label mode")
        class_to_idx = load_imagenet_synset_index(args.synset_index)
        if model.cfg.num_classes != 1000:
            raise SystemExit(f"imagenet eval needs 1000-way head, found {model.cfg.num_classes}")
    dataset = ImageFolder(os.path.join(args.data, args.split), train=False, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    modes = []
    if args.halting in ("on", "both") and not isinstance(model, PlainViT) and model.cfg.router != "none":
        modes.append(True)
    if args.halting in ("off", "both"):
        modes.append(False)
    if not modes:
        modes = [False]
    rows = []
    for halting in modes:
        for steps in [int(s) for s in args.steps.split(",") if s]:
            row = run_split(model, loader, device, steps, halting and model.cfg.router != "none", False, 0.0, 0.0)
            print(json.dumps(row), flush=True)
            rows.append(row)
        if args.adaptive and halting and model.cfg.router != "none":
            row = run_split(
                model, loader, device, max(int(s) for s in args.steps.split(",")), True, True, args.residual_tol, args.min_active
            )
            row["note"] = "adaptive"
            print(json.dumps(row), flush=True)
            rows.append(row)
    analysis = None
    if args.analyze and (not isinstance(model, PlainViT)) and model.cfg.router != "none":
        analysis = analyze_halts(model, dataset, device, max(int(s) for s in args.steps.split(",")), args.analyze, os.path.join(args.out, "halt_maps"))
        print(json.dumps({"analysis": analysis}), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "curve.json"), "w") as f:
        json.dump({"rows": rows, "analysis": analysis, "checkpoint": args.checkpoint}, f, indent=2)


if __name__ == "__main__":
    main()
