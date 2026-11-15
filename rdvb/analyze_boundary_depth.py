"""Quantify whether the ADE20K router allocates depth to semantic boundaries."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from .evaluate import load_trained
from .evaluate_segmentation_original import load_image
from .segmentation import RDVBSegmenter


def boundary_mask(target):
    valid = target >= 0
    boundary = torch.zeros_like(valid)
    pair = valid[:, 1:] & valid[:, :-1] & (target[:, 1:] != target[:, :-1])
    boundary[:, 1:] |= pair
    boundary[:, :-1] |= pair
    pair = valid[1:, :] & valid[:-1, :] & (target[1:, :] != target[:-1, :])
    boundary[1:, :] |= pair
    boundary[:-1, :] |= pair
    return boundary, valid


@torch.inference_mode()
def analyze(model, pairs, schedule, routing_mode, seed, short_side, max_long):
    torch.manual_seed(seed)
    boundary_sum = interior_sum = 0.0
    boundary_count = interior_count = 0
    boundary_deep = interior_deep = 0
    for image_path, mask_path in pairs:
        images, target, nh, nw = load_image(image_path, mask_path, short_side, max_long)
        images = images.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.backbone(
                images,
                steps=len(schedule),
                halting=True,
                capacity_schedule=schedule,
                fold_halted=True,
                routing_mode=routing_mode,
            )
        gh, gw = images.shape[-2] // 16, images.shape[-1] // 16
        ph, pw = (nh + 15) // 16, (nw + 15) // 16
        depth = output.halt_step[0, 1:].reshape(gh, gw)[:ph, :pw].cpu().float()
        patch_target = F.interpolate(
            target[None, None].float(), size=(ph, pw), mode="nearest"
        )[0, 0].long()
        boundary, valid = boundary_mask(patch_target)
        interior = valid & ~boundary
        boundary_sum += float(depth[boundary].sum())
        interior_sum += float(depth[interior].sum())
        boundary_count += int(boundary.sum())
        interior_count += int(interior.sum())
        boundary_deep += int((depth[boundary] >= 4).sum())
        interior_deep += int((depth[interior] >= 4).sum())
    boundary_mean = boundary_sum / boundary_count
    interior_mean = interior_sum / interior_count
    return {
        "routing_mode": routing_mode,
        "seed": seed,
        "images": len(pairs),
        "boundary_patches": boundary_count,
        "interior_patches": interior_count,
        "boundary_mean_updates": boundary_mean,
        "interior_mean_updates": interior_mean,
        "boundary_minus_interior": boundary_mean - interior_mean,
        "boundary_deep_fraction": boundary_deep / boundary_count,
        "interior_deep_fraction": interior_deep / interior_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--max-long", type=int, default=1024)
    parser.add_argument("--random-seeds", default="101,102,103")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone, _ = load_trained(checkpoint["source"], torch.device("cuda"))
    model = RDVBSegmenter(backbone).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    image_dir = os.path.join(args.data, "images", "validation")
    mask_dir = os.path.join(args.data, "annotations", "validation")
    pairs = [
        (os.path.join(image_dir, name), os.path.join(mask_dir, name.replace(".jpg", ".png")))
        for name in sorted(os.listdir(image_dir)) if name.endswith(".jpg")
    ]
    schedule = tuple(float(value) for value in checkpoint["schedule"])
    rows = [analyze(model, pairs, schedule, "learned", 0, args.short_side, args.max_long)]
    for seed in (int(value) for value in args.random_seeds.split(",")):
        rows.append(analyze(model, pairs, schedule, "random", seed, args.short_side, args.max_long))
    result = {"checkpoint": args.checkpoint, "schedule": schedule, "rows": rows}
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
