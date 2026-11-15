"""Forward and backward checks for the comparison baselines."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdvb.baselines import build_baseline


def test_all_baselines_train_step():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    for name in ("plain12", "early_exit", "token_prune", "tome", "mod", "swin_t"):
        model = build_baseline(name, num_classes=10).train()
        out = model(images)
        loss = out.logits.sum()
        if out.aux_logits:
            loss = loss + sum(item.sum() for item in out.aux_logits)
        loss.backward()
        assert torch.isfinite(out.logits).all(), name
        print(name, "ok", tuple(out.logits.shape), "tokens", out.tokens.shape[1])


if __name__ == "__main__":
    test_all_baselines_train_step()
    print("baselines ok")
