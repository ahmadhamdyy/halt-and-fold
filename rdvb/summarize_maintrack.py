"""Aggregate repeated experiments into JSON and Markdown summaries."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "values": [float(x) for x in array]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default="runs/maintrack_summary.json")
    args = parser.parse_args()
    segmentation = []
    for seed in range(3):
        path = os.path.join(args.root, f"ade_seed{seed}", "best.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metrics = checkpoint["metrics"]
        segmentation.append({"seed": seed, "epoch": checkpoint["epoch"], **metrics})
    capacity = []
    for seed in range(3):
        path = os.path.join(args.root, f"capacity_seed{seed}", "result.json")
        with open(path) as f:
            result = json.load(f)
        capacity.append({"seed": seed, **result})
    aggregate = {
        "segmentation": {
            "runs": segmentation,
            "frozen_kv_miou": stats([x["frozen_kv"]["miou"] for x in segmentation]),
            "dense_miou": stats([x["dense"]["miou"] for x in segmentation]),
            "drop_miou": stats([x["drop_tokens"]["miou"] for x in segmentation]),
            "frozen_minus_drop": stats([
                x["frozen_kv"]["miou"] - x["drop_tokens"]["miou"] for x in segmentation
            ]),
        },
        "capacity_router": {"runs": capacity, "top1": stats([x["best"] for x in capacity])},
    }
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
