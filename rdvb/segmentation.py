"""ADE20K semantic-segmentation transfer for RDVB.

The pipeline keeps every output patch so frozen-K/V routing and token dropping
can be compared without changing the decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_trained


class ADE20K(Dataset):
    def __init__(self, root: str, split: str, crop: int = 512):
        self.image_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "annotations", split)
        self.train = split == "training"
        self.crop = crop
        self.images = sorted(
            os.path.join(self.image_dir, name)
            for name in os.listdir(self.image_dir)
            if name.lower().endswith(".jpg")
        )
        if not self.images:
            raise FileNotFoundError(f"no ADE20K images in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        with Image.open(image_path) as item:
            image = item.convert("RGB")
        with Image.open(mask_path) as item:
            mask = item.copy()
        if self.train:
            target_short = random.randint(int(self.crop * 0.75), int(self.crop * 1.5))
            w, h = image.size
            scale = target_short / min(w, h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            image = image.resize((nw, nh), Image.BICUBIC)
            mask = mask.resize((nw, nh), Image.NEAREST)
            pad_w, pad_h = max(0, self.crop - nw), max(0, self.crop - nh)
            if pad_w or pad_h:
                canvas = Image.new("RGB", (nw + pad_w, nh + pad_h))
                canvas.paste(image, (0, 0))
                image = canvas
                target = Image.new("L", (nw + pad_w, nh + pad_h), color=0)
                target.paste(mask, (0, 0))
                mask = target
            left = random.randint(0, image.width - self.crop)
            top = random.randint(0, image.height - self.crop)
            box = (left, top, left + self.crop, top + self.crop)
            image, mask = image.crop(box), mask.crop(box)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image = image.resize((self.crop, self.crop), Image.BICUBIC)
            mask = mask.resize((self.crop, self.crop), Image.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        image_t = (image_t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        # ADEChallenge masks use 0=ignore and 1..150=semantic classes.
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()) - 1
        return image_t, mask_t


class RDVBSegmenter(nn.Module):
    def __init__(self, backbone, classes: int = 150):
        super().__init__()
        self.backbone = backbone
        dim = backbone.cfg.embed_dim
        self.decode = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, classes, 1),
        )

    def forward(
        self,
        images,
        steps=None,
        halting=True,
        schedule=None,
        impl="auto",
        output_size=None,
        fold_halted=False,
        fold_writeback="final",
        routing_mode="learned",
    ):
        steps = len(schedule) if steps is None and schedule is not None else (steps or 4)
        out = self.backbone(
            images,
            steps=steps,
            halting=halting,
            capacity_schedule=schedule,
            impl=impl,
            fold_halted=fold_halted,
            fold_writeback=fold_writeback,
            routing_mode=routing_mode,
        )
        patch = self.backbone.norm(out.tokens[:, 1:])
        gh = images.shape[-2] // self.backbone.cfg.patch_size
        gw = images.shape[-1] // self.backbone.cfg.patch_size
        feature = patch.transpose(1, 2).reshape(images.shape[0], -1, gh, gw)
        logits = self.decode(feature)
        if output_size is None:
            output_size = images.shape[-2:]
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits, out


def confusion_update(confusion, pred, target, classes=150):
    valid = (target >= 0) & (target < classes)
    encoded = target[valid] * classes + pred[valid]
    confusion += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes).cpu()


@torch.inference_mode()
def evaluate(
    model, loader, schedule, kv_mode, steps=None, fold_halted=False, fold_writeback="final"
):
    model.eval()
    old = model.backbone.cfg.kv_mode
    model.backbone.cfg.kv_mode = kv_mode
    confusion = torch.zeros(150, 150, dtype=torch.int64)
    ponder = count = 0.0
    for images, target in loader:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, out = model(
                images,
                steps=steps,
                halting=schedule is not None,
                schedule=schedule,
                fold_halted=fold_halted,
                fold_writeback=fold_writeback,
            )
        confusion_update(confusion, logits.argmax(1), target)
        ponder += float(out.halt_step[:, 1:].float().mean(1).sum())
        count += images.shape[0]
    model.backbone.cfg.kv_mode = old
    tp = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - tp
    valid = union > 0
    iou = tp[valid] / union[valid]
    pixel = tp.sum() / confusion.sum().clamp_min(1)
    return {"miou": float(iou.mean()), "pixel_acc": float(pixel), "patch_updates": ponder / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--loss-size", type=int, default=128,
                        help="train the decoder loss at this resolution; validation remains full crop size")
    parser.add_argument("--schedule", default="1,.75,.5,.25")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    schedule = tuple(float(x) for x in args.schedule.split(","))
    backbone, source = load_trained(args.checkpoint, torch.device("cuda"))
    backbone.head.requires_grad_(False)
    model = RDVBSegmenter(backbone).cuda()
    resume = None
    start_epoch = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        start_epoch = int(resume["epoch"]) + 1
    train_set = ADE20K(args.data, "training", args.crop)
    val_set = ADE20K(args.data, "validation", args.crop)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    backbone_params = [p for name, p in model.backbone.named_parameters() if p.requires_grad and not name.startswith("head")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr}, {"params": model.decode.parameters(), "lr": args.lr * 10}],
        weight_decay=0.01,
    )
    if resume is not None and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
    total_steps = args.epochs * len(train_loader)
    global_step = start_epoch * len(train_loader)
    best = float(resume["metrics"]["frozen_kv"]["miou"]) if resume is not None else -1.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        start = time.time()
        for images, target in train_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            progress = global_step / max(1, total_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = group.get("initial_lr", group["lr"]) if global_step == 0 else group["lr"]
            optimizer.param_groups[0]["lr"] = args.lr * scale
            optimizer.param_groups[1]["lr"] = args.lr * 10 * scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_hw = (args.loss_size, args.loss_size)
                logits, _ = model(images, halting=True, schedule=schedule, output_size=loss_hw)
                target_low = F.interpolate(target.unsqueeze(1).float(), size=loss_hw, mode="nearest").squeeze(1).long()
                loss = F.cross_entropy(logits, target_low, ignore_index=-1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            global_step += 1
        frozen = evaluate(model, val_loader, schedule, "frozen", steps=len(schedule))
        dense = evaluate(model, val_loader, None, "frozen", steps=len(schedule))
        dropped = evaluate(model, val_loader, schedule, "drop", steps=len(schedule))
        row = {"epoch": epoch, "loss": sum(losses) / len(losses), "seconds": time.time() - start,
               "frozen_kv": frozen, "dense": dense, "drop_tokens": dropped}
        print(json.dumps(row), flush=True)
        with open(os.path.join(args.out, "train.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        checkpoint = {"model": model.state_dict(), "backbone_cfg": model.backbone.cfg.to_dict(),
                      "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": row,
                      "schedule": schedule, "source": args.checkpoint}
        torch.save(checkpoint, os.path.join(args.out, "last.pt"))
        if frozen["miou"] > best:
            best = frozen["miou"]
            torch.save(checkpoint, os.path.join(args.out, "best.pt"))
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"best_frozen_kv_miou": best, "source": args.checkpoint, "schedule": schedule}, f, indent=2)


if __name__ == "__main__":
    main()
