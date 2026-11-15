"""Learned-versus-random ADE20K routing at a matched capacity schedule."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from .evaluate import load_trained
from .evaluate_segmentation_original import evaluate
from .segmentation import RDVBSegmenter


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
    torch.manual_seed(0)
    learned = evaluate(
        model, pairs, schedule, "frozen", args.short_side, args.max_long,
        steps=len(schedule), fold_halted=True, routing_mode="learned"
    )
    random_rows = []
    for seed in (int(value) for value in args.random_seeds.split(",")):
        torch.manual_seed(seed)
        row = evaluate(
            model, pairs, schedule, "frozen", args.short_side, args.max_long,
            steps=len(schedule), fold_halted=True, routing_mode="random"
        )
        row["seed"] = seed
        random_rows.append(row)
    random_miou = [row["miou"] for row in random_rows]
    result = {
        "checkpoint": args.checkpoint,
        "schedule": schedule,
        "learned": learned,
        "random_rows": random_rows,
        "random_miou_mean": statistics.mean(random_miou),
        "random_miou_std": statistics.stdev(random_miou),
        "learned_minus_random_miou": learned["miou"] - statistics.mean(random_miou),
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
