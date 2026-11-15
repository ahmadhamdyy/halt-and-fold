"""Render qualitative ADE20K recurrent-depth maps."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .evaluate import load_trained
from .evaluate_segmentation_original import load_image
from .segmentation import RDVBSegmenter


PALETTE = np.asarray(
    [
        (68, 1, 84),
        (70, 50, 126),
        (54, 92, 141),
        (39, 127, 142),
        (31, 161, 135),
        (74, 193, 109),
        (160, 218, 57),
        (253, 231, 37),
    ],
    dtype=np.uint8,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--names", default="ADE_val_00000001,ADE_val_00000076")
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--max-long", type=int, default=1024)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone, _ = load_trained(checkpoint["source"], torch.device("cuda"))
    model = RDVBSegmenter(backbone).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    schedule = tuple(float(x) for x in checkpoint["schedule"])

    rows = []
    for stem in args.names.split(","):
        image_path = os.path.join(args.data, "images", "validation", stem + ".jpg")
        mask_path = os.path.join(args.data, "annotations", "validation", stem + ".png")
        images, _, nh, nw = load_image(
            image_path, mask_path, args.short_side, args.max_long
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, output = model(
                images.cuda(),
                steps=len(schedule),
                halting=True,
                schedule=schedule,
                fold_halted=True,
            )
        gh, gw = images.shape[-2] // 16, images.shape[-1] // 16
        depth = output.halt_step[0, 1:].reshape(gh, gw).cpu().numpy()
        depth = np.clip(depth[: (nh + 15) // 16, : (nw + 15) // 16], 1, 8)
        color = Image.fromarray(PALETTE[depth - 1]).resize((nw, nh), Image.Resampling.NEAREST)
        with Image.open(image_path) as item:
            image = item.convert("RGB").resize((nw, nh), Image.Resampling.BICUBIC)
        overlay = Image.blend(image, color, 0.48)
        rows.append((stem, image, overlay))

    panel_w = 420
    header_h = 52
    row_h = 245
    canvas = Image.new("RGB", (2 * panel_w, header_h + len(rows) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((panel_w // 2 - 22, 8), "Input", fill="black")
    draw.text((panel_w + panel_w // 2 - 58, 8), "Recurrent depth", fill="black")
    for step, rgb in enumerate(PALETTE, 1):
        x0 = panel_w + 48 + (step - 1) * 42
        draw.rectangle((x0, 28, x0 + 30, 42), fill=tuple(int(v) for v in rgb))
        draw.text((x0 + 10, 42), str(step), fill="black")

    for row, (stem, image, overlay) in enumerate(rows):
        y0 = header_h + row * row_h
        scale = min((panel_w - 12) / image.width, (row_h - 24) / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        left = image.resize(size, Image.Resampling.LANCZOS)
        right = overlay.resize(size, Image.Resampling.LANCZOS)
        x_pad = (panel_w - size[0]) // 2
        y_pad = (row_h - size[1]) // 2
        canvas.paste(left, (x_pad, y0 + y_pad))
        canvas.paste(right, (panel_w + x_pad, y0 + y_pad))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out, quality=95)


if __name__ == "__main__":
    main()
