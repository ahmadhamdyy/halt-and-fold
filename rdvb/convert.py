"""Map a DeiT / ViT checkpoint into a recurrent-depth backbone.

Entry is block 0, exit is the last block, and the shared core is either the
mean of the middle blocks or the single middle block. Iteration conditioning
is zero at initialization and the residual gate is near one, so one core
application starts as that copied block.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch

from .model import PRESETS, PlainViT, RDVB, RDVBConfig


def _state_dict(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"unrecognized checkpoint layout in {path}")


def _block_indices(state: dict[str, torch.Tensor]) -> list[int]:
    indices = set()
    for key in state:
        if key.startswith("blocks."):
            indices.add(int(key.split(".")[1]))
    return sorted(indices)


def _load_block(state: dict[str, torch.Tensor], indices: Iterable[int], module: torch.nn.Module, how: str) -> None:
    indices = list(indices)
    dst = module.state_dict()
    mapped = {}
    for key in dst:
        parts = [state[f"blocks.{i}.{key}"] for i in indices]
        if how == "mean":
            mapped[key] = sum(parts) / float(len(parts))
        elif how == "copy":
            if len(indices) != 1:
                raise ValueError("copy expects one source block")
            mapped[key] = parts[0]
        else:
            raise ValueError(how)
    module.load_state_dict(mapped)


def convert_deit(deit_path: str, how: str = "mean", router: str = "spatial", router_bias: float = -2.0) -> RDVB:
    state = _state_dict(deit_path)
    indices = _block_indices(state)
    if len(indices) < 3:
        raise ValueError(f"need at least 3 ViT blocks, found {len(indices)}")
    dim = state["pos_embed"].shape[-1]
    num_classes = state["head.weight"].shape[0]
    preset = next((name for name, cfg in PRESETS.items() if cfg["embed_dim"] == dim), None)
    heads = PRESETS[preset]["num_heads"] if preset else {192: 3, 384: 6, 768: 12}[dim]
    cfg = RDVBConfig(
        embed_dim=dim,
        num_heads=heads,
        num_classes=num_classes,
        router=router,
        router_bias=router_bias,
        kv_mode="frozen",
    )
    model = RDVB(cfg)
    model.patch_embed.proj.weight.data.copy_(state["patch_embed.proj.weight"])
    model.patch_embed.proj.bias.data.copy_(state["patch_embed.proj.bias"])
    model.cls_token.data.copy_(state["cls_token"])
    model.pos_embed.data.copy_(state["pos_embed"])
    model.norm.weight.data.copy_(state["norm.weight"])
    model.norm.bias.data.copy_(state["norm.bias"])
    model.head.weight.data.copy_(state["head.weight"])
    model.head.bias.data.copy_(state["head.bias"])
    _load_block(state, [indices[0]], model.entry[0], "copy")
    middle = indices[1:-1]
    if how == "mean":
        _load_block(state, middle, model.core.block, "mean")
    elif how == "middle":
        _load_block(state, [middle[len(middle) // 2]], model.core.block, "copy")
    else:
        raise ValueError(how)
    _load_block(state, [indices[-1]], model.exit[0], "copy")
    model.core.reset_conditioning()
    if model.router is not None:
        model.router.reset_parameters(router_bias)
    return model


def deit_as_plain(deit_path: str) -> PlainViT:
    """Reference stack: the original DeiT with every block left unique."""
    state = _state_dict(deit_path)
    indices = _block_indices(state)
    dim = state["pos_embed"].shape[-1]
    num_classes = state["head.weight"].shape[0]
    preset = next((name for name, cfg in PRESETS.items() if cfg["embed_dim"] == dim), None)
    heads = PRESETS[preset]["num_heads"] if preset else {192: 3, 384: 6, 768: 12}[dim]
    cfg = RDVBConfig(embed_dim=dim, num_heads=heads, num_classes=num_classes, router="none")
    model = PlainViT(cfg, depth=len(indices))
    model.patch_embed.proj.weight.data.copy_(state["patch_embed.proj.weight"])
    model.patch_embed.proj.bias.data.copy_(state["patch_embed.proj.bias"])
    model.cls_token.data.copy_(state["cls_token"])
    model.pos_embed.data.copy_(state["pos_embed"])
    model.norm.weight.data.copy_(state["norm.weight"])
    model.norm.bias.data.copy_(state["norm.bias"])
    model.head.weight.data.copy_(state["head.weight"])
    model.head.bias.data.copy_(state["head.bias"])
    for index in indices:
        _load_block(state, [index], model.blocks[index], "copy")
    return model


def save_converted(model: RDVB, path: str, source: str, how: str) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "source": source,
            "init": how,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", choices=("mean", "middle", "plain"), default="mean")
    parser.add_argument("--router", choices=("spatial", "independent", "none"), default="spatial")
    args = parser.parse_args()
    if args.init == "plain":
        model = deit_as_plain(args.deit)
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": model.cfg.to_dict(),
                "arch": "plain",
                "plain_depth": model.depth,
                "source": args.deit,
                "init": "plain",
            },
            args.out,
        )
        n = sum(p.numel() for p in model.parameters())
        print(f"wrote {args.out} params={n/1e6:.2f}M init=plain depth={model.depth}")
        return
    model = convert_deit(args.deit, how=args.init, router=args.router)
    save_converted(model, args.out, args.deit, args.init)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out} params={n/1e6:.2f}M init={args.init} dim={model.cfg.embed_dim}")


if __name__ == "__main__":
    main()
