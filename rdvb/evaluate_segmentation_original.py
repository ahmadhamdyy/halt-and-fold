"""Aspect-preserving single-scale ADE20K evaluation for a segmenter checkpoint."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_trained
from .segmentation import RDVBSegmenter, confusion_update


def load_image(image_path, mask_path, short_side, max_long):
    with Image.open(image_path) as item:
        image = item.convert("RGB")
    with Image.open(mask_path) as item:
        mask = item.copy()
    w, h = image.size
    scale = short_side / min(w, h)
    if max(w, h) * scale > max_long:
        scale = max_long / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    # Patch embedding requires dimensions divisible by 16. Padding is cropped
    # out of logits before scoring.
    pw, ph = ((nw + 15) // 16) * 16, ((nh + 15) // 16) * 16
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (pw, ph))
    canvas.paste(resized, (0, 0))
    target = mask.resize((nw, nh), Image.NEAREST)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    tensor = (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    target = torch.from_numpy(np.asarray(target, dtype=np.int64).copy()) - 1
    return tensor.unsqueeze(0), target, nh, nw


@torch.inference_mode()
def evaluate(
    model,
    pairs,
    schedule,
    mode,
    short_side,
    max_long,
    steps=None,
    fold_halted=False,
    fold_writeback="final",
    routing_mode="learned",
):
    model.eval()
    model.backbone.cfg.kv_mode = mode
    confusion = torch.zeros(150, 150, dtype=torch.int64)
    ponder = 0.0
    for image_path, mask_path in pairs:
        images, target, nh, nw = load_image(image_path, mask_path, short_side, max_long)
        images = images.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, out = model(
                images,
                steps=steps,
                halting=schedule is not None,
                schedule=schedule,
                fold_halted=fold_halted,
                fold_writeback=fold_writeback,
                routing_mode=routing_mode,
            )
        prediction = logits[:, :, :nh, :nw].argmax(1).cpu()[0]
        confusion_update(confusion, prediction, target)
        ponder += float(out.halt_step[:, 1:].float().mean())
    tp = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - tp
    valid = union > 0
    return {
        "miou": float((tp[valid] / union[valid]).mean()),
        "pixel_acc": float(tp.sum() / confusion.sum().clamp_min(1)),
        "patch_updates": ponder / len(pairs),
        "images": len(pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--max-long", type=int, default=1024)
    parser.add_argument("--schedule", default="1,.75,.5,.25")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone, _ = load_trained(checkpoint["source"], torch.device("cuda"))
    model = RDVBSegmenter(backbone).cuda()
    model.load_state_dict(checkpoint["model"])
    image_dir = os.path.join(args.data, "images", "validation")
    mask_dir = os.path.join(args.data, "annotations", "validation")
    pairs = []
    for name in sorted(os.listdir(image_dir)):
        if name.endswith(".jpg"):
            pairs.append((os.path.join(image_dir, name), os.path.join(mask_dir, name.replace(".jpg", ".png"))))
    schedule = tuple(float(x) for x in args.schedule.split(","))
    result = {
        "protocol": {"resize_short_side": args.short_side, "max_long_side": args.max_long, "aspect_preserving": True},
        "frozen_kv": evaluate(
            model, pairs, schedule, "frozen", args.short_side, args.max_long, steps=len(schedule)
        ),
        "halt_and_fold": evaluate(
            model,
            pairs,
            schedule,
            "frozen",
            args.short_side,
            args.max_long,
            steps=len(schedule),
            fold_halted=True,
        ),
        "stale_writeback": evaluate(
            model,
            pairs,
            schedule,
            "frozen",
            args.short_side,
            args.max_long,
            steps=len(schedule),
            fold_halted=True,
            fold_writeback="stale",
        ),
        "dense": evaluate(
            model, pairs, None, "frozen", args.short_side, args.max_long, steps=len(schedule)
        ),
        "drop_tokens": evaluate(
            model, pairs, schedule, "drop", args.short_side, args.max_long, steps=len(schedule)
        ),
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

