"""Recurrent-depth vision backbone.

A ViT with a small unique entry, a weight-shared recurrent core, and a small
unique exit. The core is applied a variable number of times. A router halts
easy tokens; halted tokens stay in the sequence as frozen keys and values and
are not updated as queries.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ViTBlock
from .sparse import ste_mask


@dataclass
class RDVBConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 384
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0
    depth_entry: int = 1
    depth_exit: int = 1
    # spatial: halt logit sees a 3x3 neighborhood. independent: token only. none: no halting.
    router: str = "spatial"
    kv_mode: str = "frozen"
    router_bias: float = -1.5
    halt_threshold: float = 0.5
    # ``film`` gives the shared block depth-specific normalization and
    # residual modulation while retaining a single set of block weights.
    core_mode: str = "additive"
    cond_dim: int = 128
    # Entry/exit retain softmax attention; the shared core may use normalized
    # linear attention to enable exact halt-and-fold sufficient statistics.
    core_attention: str = "softmax"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "RDVBConfig":
        known = {k: data[k] for k in RDVBConfig.__dataclass_fields__ if k in data}
        return RDVBConfig(**known)


PRESETS = {
    "ti": dict(embed_dim=192, num_heads=3),
    "s": dict(embed_dim=384, num_heads=6),
    "b": dict(embed_dim=768, num_heads=12),
}


class ForwardOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        aux_logits: list[torch.Tensor],
        ponder_soft: torch.Tensor,
        ponder_hard: torch.Tensor,
        halt_step: torch.Tensor,
        tokens: torch.Tensor,
        steps_run: int,
    ):
        self.logits = logits
        self.aux_logits = aux_logits
        self.ponder_soft = ponder_soft
        self.ponder_hard = ponder_hard
        self.halt_step = halt_step
        self.tokens = tokens
        self.steps_run = steps_run


def fourier_step(step: int, dim: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal embedding of a recurrence index. Defined for every integer step."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1))
    ang = float(step) * freqs
    emb = torch.cat([ang.sin(), ang.cos()], dim=0)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def interpolate_pos_embed(pos: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    cls, grid = pos[:, :1], pos[:, 1:]
    gs = int(round(math.sqrt(grid.shape[1])))
    if gs * gs != grid.shape[1]:
        raise RuntimeError("positional grid is not square")
    if gs == gh and gs == gw:
        return pos
    c = grid.shape[-1]
    grid = grid.reshape(1, gs, gs, c).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, c)
    return torch.cat([cls, grid], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class HaltRouter(nn.Module):
    def __init__(self, dim: int, kind: str, bias: float):
        super().__init__()
        if kind not in ("spatial", "independent"):
            raise ValueError(kind)
        self.kind = kind
        self.dw = None
        if kind == "spatial":
            self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
            self.fc = nn.Linear(dim * 2, 1)
        else:
            self.fc = nn.Linear(dim, 1)
        self.reset_parameters(bias)

    def reset_parameters(self, bias: float) -> None:
        if self.dw is not None:
            nn.init.constant_(self.dw.weight, 1.0 / 9.0)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, bias)

    def forward(self, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        patches = tokens[:, 1:]
        if self.kind == "spatial":
            b, _, c = patches.shape
            feat = patches.transpose(1, 2).reshape(b, c, gh, gw)
            neigh = self.dw(feat).flatten(2).transpose(1, 2)
            inp = torch.cat([patches, neigh], dim=-1)
        else:
            inp = patches
        patch_logits = self.fc(inp).squeeze(-1)
        cls_logits = tokens.new_full((tokens.shape[0], 1), -30.0)
        return torch.cat([cls_logits, patch_logits], dim=1)


class RecurrentCore(nn.Module):
    """Shared block plus a learned per-step gate.

    Fourier features of the step index go through an MLP whose last layer is
    zero-initialized, and the residual gate starts near 1. At initialization,
    one application matches the underlying pretrained block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        mode: str = "additive",
        cond_dim: int = 128,
        attention_kind: str = "softmax",
    ):
        super().__init__()
        if mode not in ("additive", "film"):
            raise ValueError(f"unknown recurrent core mode {mode}")
        self.mode = mode
        self.block = ViTBlock(dim, num_heads, mlp_ratio, drop, attention_kind=attention_kind)
        hidden = dim if mode == "additive" else cond_dim
        output = dim if mode == "additive" else 6 * dim
        self.cond = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, output))
        self.gate = nn.Linear(dim, dim) if mode == "additive" else None
        self.reset_conditioning()

    def reset_conditioning(self) -> None:
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 4.0)

    def condition(self, step: int, ref: torch.Tensor):
        freq = fourier_step(step, ref.shape[-1], ref.device).to(dtype=ref.dtype)
        cond = self.cond(freq)
        if self.mode == "additive":
            gate = torch.sigmoid(self.gate(cond))
            return cond, gate
        scale_a, shift_a, scale_m, shift_m, raw_gate_a, raw_gate_m = cond.chunk(6)
        # tanh bounds FiLM excursions.  2*sigmoid(0)=1, so a newly created
        # recurrent core is exactly the underlying unmodulated block.
        return (
            torch.tanh(scale_a),
            torch.tanh(shift_a),
            torch.tanh(scale_m),
            torch.tanh(shift_m),
            2.0 * torch.sigmoid(raw_gate_a),
            2.0 * torch.sigmoid(raw_gate_m),
        )

    def forward(
        self,
        x: torch.Tensor,
        step: int,
        ste: torch.Tensor | None,
        hard: torch.Tensor | None,
        kv_mode: str,
        impl: str = "auto",
        capacity: int | None = None,
    ) -> torch.Tensor:
        conditioned = self.condition(step, x)
        if self.mode == "film":
            if hard is None:
                return self.block(x, modulation=conditioned)
            y = self.block(
                x,
                active=hard,
                kv_mode=kv_mode,
                impl=impl,
                modulation=conditioned,
                capacity=capacity,
            )
            # The block enforces exact hard masking in the forward pass; the
            # straight-through survival probability trains the router.
            return x + ste.unsqueeze(-1).to(dtype=x.dtype) * (y - x)
        cond, gate = conditioned
        if hard is None:
            y = self.block(x + cond.view(1, 1, -1))
            return x + gate.view(1, 1, -1) * (y - x)
        cond_tok = cond.view(1, 1, -1) * hard.unsqueeze(-1).to(dtype=x.dtype)
        y = self.block(x + cond_tok, active=hard, kv_mode=kv_mode, impl=impl, capacity=capacity)
        return x + ste.unsqueeze(-1).to(dtype=x.dtype) * gate.view(1, 1, -1) * (y - x)

    def forward_cached(
        self,
        x: torch.Tensor,
        step: int,
        ste: torch.Tensor,
        hard: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None,
        capacity: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Exact cached execution for the additive recurrent core."""
        if self.mode != "additive":
            raise ValueError("cross-depth K/V caching currently requires additive core conditioning")
        cond, gate = self.condition(step, x)
        cond_tok = cond.view(1, 1, -1) * hard.unsqueeze(-1).to(dtype=x.dtype)
        y, cache = self.block.forward_cached(x + cond_tok, hard, cache, capacity)
        x = x + ste.unsqueeze(-1).to(dtype=x.dtype) * gate.view(1, 1, -1) * (y - x)
        return x, cache

    def forward_folded(
        self,
        x: torch.Tensor,
        step: int,
        ste: torch.Tensor,
        hard: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None,
        capacity: int | None = None,
        stale_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        if self.mode != "additive":
            raise ValueError("halt-and-fold currently requires additive core conditioning")
        cond, gate = self.condition(step, x)
        cond_tok = cond.view(1, 1, -1) * hard.unsqueeze(-1).to(dtype=x.dtype)
        writeback_source = None
        if stale_state is not None and memory is not None:
            previous_active = memory[2]
            previous_cond, _ = self.condition(step - 1, stale_state)
            previous_cond_tok = previous_cond.view(1, 1, -1) * previous_active.unsqueeze(-1).to(
                dtype=stale_state.dtype
            )
            # Stale-writeback ablation: use the state before the final update.
            writeback_source = self.block.norm1(stale_state + previous_cond_tok)
        y, memory = self.block.forward_folded(
            x + cond_tok,
            hard,
            memory,
            capacity,
            writeback_source=writeback_source,
        )
        x = x + ste.unsqueeze(-1).to(dtype=x.dtype) * gate.view(1, 1, -1) * (y - x)
        return x, memory


class RDVB(nn.Module):
    def __init__(self, cfg: RDVBConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, dim)
        grid = cfg.img_size // cfg.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, dim))
        self.pos_drop = nn.Dropout(cfg.drop)
        self.entry = nn.ModuleList(
            [ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(cfg.depth_entry)]
        )
        self.core = RecurrentCore(
            dim,
            cfg.num_heads,
            cfg.mlp_ratio,
            cfg.drop,
            mode=cfg.core_mode,
            cond_dim=cfg.cond_dim,
            attention_kind=cfg.core_attention,
        )
        self.exit = nn.ModuleList(
            [ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(cfg.depth_exit)]
        )
        self.router = None
        if cfg.router != "none":
            self.router = HaltRouter(dim, cfg.router, cfg.router_bias)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, cfg.num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.router is not None:
            self.router.reset_parameters(cfg.router_bias)
        self.core.reset_conditioning()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def classify(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(tokens[:, 0]))

    def embed(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        patches = self.patch_embed(images)
        b, n, _ = patches.shape
        ps = self.cfg.patch_size
        gh, gw = images.shape[-2] // ps, images.shape[-1] // ps
        if gh * gw != n:
            raise RuntimeError(f"patch grid {gh}x{gw} does not match {n} tokens")
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + interpolate_pos_embed(self.pos_embed, gh, gw)
        return self.pos_drop(tokens), gh, gw

    def _exit(
        self,
        x: torch.Tensor,
        ste: torch.Tensor | None,
        hard: torch.Tensor | None,
        impl: str,
        capacity: int | None = None,
    ) -> torch.Tensor:
        for blk in self.exit:
            if hard is None:
                x = blk(x)
            else:
                y = blk(x, active=hard, kv_mode=self.cfg.kv_mode, impl=impl, capacity=capacity)
                x = x + ste.unsqueeze(-1).to(dtype=x.dtype) * (y - x)
        return x

    def forward(
        self,
        images: torch.Tensor,
        steps: int = 4,
        halting: bool | None = None,
        adaptive: bool = False,
        residual_tol: float = 0.0,
        min_active: float = 0.0,
        fixed_active: torch.Tensor | None = None,
        impl: str = "auto",
        capacity_schedule: tuple[float, ...] | list[float] | None = None,
        cache_kv: bool = False,
        fold_halted: bool = False,
        fold_writeback: str = "final",
        routing_mode: str = "learned",
    ) -> ForwardOutput:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if capacity_schedule is not None:
            if len(capacity_schedule) < steps:
                raise ValueError("capacity_schedule must provide one fraction per recurrent step")
            if any(not (0.0 < float(v) <= 1.0) for v in capacity_schedule[:steps]):
                raise ValueError("capacity fractions must be in (0, 1]")
        if cache_kv and self.cfg.kv_mode != "frozen":
            raise ValueError("cross-depth K/V caching requires kv_mode='frozen'")
        if cache_kv and self.cfg.core_mode != "additive":
            raise ValueError("cross-depth K/V caching currently requires core_mode='additive'")
        if cache_kv and self.training:
            raise ValueError("cross-depth K/V caching is an inference-only execution path")
        if fold_halted and cache_kv:
            raise ValueError("choose either projection caching or halt-and-fold execution")
        if fold_halted and self.cfg.core_attention != "linear":
            raise ValueError("halt-and-fold requires core_attention='linear'")
        if fold_halted and self.cfg.kv_mode != "frozen":
            raise ValueError("halt-and-fold requires kv_mode='frozen'")
        if fold_halted and self.cfg.core_mode != "additive":
            raise ValueError("halt-and-fold currently requires core_mode='additive'")
        if fold_halted and self.training:
            raise ValueError("halt-and-fold is an inference-only execution path")
        if fold_writeback not in ("final", "stale"):
            raise ValueError("fold_writeback must be 'final' or 'stale'")
        if fold_writeback != "final" and not fold_halted:
            raise ValueError("non-final write-back requires fold_halted=True")
        if routing_mode not in ("learned", "random"):
            raise ValueError("routing_mode must be 'learned' or 'random'")
        cfg = self.cfg
        use_halt = cfg.router != "none" if halting is None else (halting and cfg.router != "none")
        if use_halt and self.router is None and fixed_active is None:
            raise RuntimeError("halting requested but this model has no router")

        x, gh, gw = self.embed(images)
        for blk in self.entry:
            x = blk(x)

        b, n, _ = x.shape
        device = x.device
        hard = torch.ones(b, n, dtype=torch.bool, device=device)
        soft = x.new_ones(b, n)
        if fixed_active is not None:
            hard = fixed_active.bool().clone()
            hard[:, 0] = True
            soft = hard.float()
        halt_step = torch.full((b, n), -1, dtype=torch.long, device=device)
        ponder_soft_acc = x.new_zeros(b, n)
        aux: list[torch.Tensor] = []
        executed = 0
        active_capacity = n
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        folded_memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        previous_fold_input: torch.Tensor | None = None

        for t in range(steps):
            if use_halt or fixed_active is not None:
                ste = ste_mask(hard.float(), soft)
                x_prev = x
                if fold_halted:
                    fold_capacity = (
                        active_capacity if capacity_schedule is not None or fixed_active is not None else None
                    )
                    x, folded_memory = self.core.forward_folded(
                        x,
                        t,
                        ste,
                        hard,
                        folded_memory,
                        fold_capacity,
                        stale_state=previous_fold_input if fold_writeback == "stale" else None,
                    )
                    previous_fold_input = x_prev
                elif cache_kv:
                    cache_capacity = (
                        active_capacity if capacity_schedule is not None or fixed_active is not None else None
                    )
                    x, kv_cache = self.core.forward_cached(
                        x, t, ste, hard, kv_cache, cache_capacity
                    )
                else:
                    x = self.core(x, t, ste, hard, cfg.kv_mode, impl, active_capacity)
            else:
                x_prev = x
                x = self.core(x, t, None, None, cfg.kv_mode, impl)
            executed = t + 1
            aux.append(self.classify(x))
            ponder_soft_acc = ponder_soft_acc + soft

            if not use_halt and fixed_active is None:
                continue
            if fixed_active is not None:
                # Mask is held fixed so the benchmark measures steady-state sparsity.
                continue
            if t == steps - 1 and not adaptive:
                continue

            raw = torch.sigmoid(self.router(x, gh, gw)).clamp(1e-4, 1.0 - 1e-4)
            if routing_mode == "random":
                raw = torch.rand_like(raw).clamp(1e-4, 1.0 - 1e-4)
            if capacity_schedule is not None:
                # The schedule describes the fraction active *on each update*.
                # Select the next update's absolute capacity using the learned
                # halt score while preventing previously halted patches from
                # re-entering. The class token is handled separately.
                next_fraction = float(capacity_schedule[min(t + 1, steps - 1)])
                next_patches = max(1, min(n - 1, int(round(next_fraction * (n - 1)))))
                continue_score = -raw[:, 1:]
                continue_score = continue_score.masked_fill(~hard[:, 1:], torch.finfo(continue_score.dtype).min)
                chosen = continue_score.topk(next_patches, dim=1, sorted=False).indices
                next_hard = torch.zeros_like(hard[:, 1:])
                next_hard.scatter_(1, chosen, True)
                newly = hard[:, 1:] & ~next_hard
                halt_step[:, 1:] = torch.where(
                    newly,
                    torch.full_like(halt_step[:, 1:], t + 1),
                    halt_step[:, 1:],
                )
                hard = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=device), next_hard], dim=1)
                # Hard top-k controls capacity; the keep probability supplies
                # a straight-through gradient to the router.
                keep_soft = (1.0 - raw[:, 1:]) * hard[:, 1:].float()
                patch_soft = ste_mask(next_hard.float(), keep_soft)
                soft = torch.cat([patch_soft.new_ones(b, 1), patch_soft], dim=1)
                active_capacity = next_patches + 1
                continue
            p = torch.where(hard, raw, torch.ones_like(raw))
            p = torch.cat([p.new_zeros(b, 1), p[:, 1:]], dim=1)
            newly = (p > cfg.halt_threshold) & hard
            hard = hard & ~newly
            hard = torch.cat(
                [torch.ones(b, 1, dtype=torch.bool, device=device), hard[:, 1:]],
                dim=1,
            )
            soft = soft * (1.0 - p)
            soft = torch.cat([soft.new_ones(b, 1), soft[:, 1:]], dim=1)
            halt_step = torch.where(newly, torch.full_like(halt_step, t + 1), halt_step)

            if adaptive and (not self.training) and (residual_tol > 0 or min_active > 0):
                resid = (x - x_prev).float().pow(2).mean(dim=(1, 2)).sqrt()
                frac = hard[:, 1:].float().mean(dim=1)
                stop = frac <= min_active
                if residual_tol > 0:
                    stop = stop | (resid < residual_tol)
                if bool(stop.any()):
                    forced = stop[:, None] & hard
                    forced = torch.cat(
                        [torch.zeros(b, 1, dtype=torch.bool, device=device), forced[:, 1:]],
                        dim=1,
                    )
                    halt_step = torch.where(forced, torch.full_like(halt_step, t + 1), halt_step)
                    keep = ~stop[:, None]
                    hard = hard & keep
                    hard = torch.cat(
                        [torch.ones(b, 1, dtype=torch.bool, device=device), hard[:, 1:]],
                        dim=1,
                    )
                    soft = torch.where(stop[:, None], torch.zeros_like(soft), soft)
                    soft = torch.cat([soft.new_ones(b, 1), soft[:, 1:]], dim=1)
            if use_halt and (not self.training) and adaptive and (not bool(hard[:, 1:].any())):
                break

        if use_halt or fixed_active is not None:
            ste = ste_mask(hard.float(), soft)
            x = self._exit(x, ste, hard, impl, active_capacity)
        else:
            x = self._exit(x, None, None, impl)

        if fixed_active is not None:
            ran = hard.long() * executed
            halt_step = ran
        else:
            still = halt_step < 0
            halt_step = torch.where(still, torch.full_like(halt_step, executed), halt_step)
        ponder_hard = halt_step.float().mean(dim=1)
        ponder_soft = ponder_soft_acc.mean(dim=1)
        return ForwardOutput(
            logits=self.classify(x),
            aux_logits=aux,
            ponder_soft=ponder_soft,
            ponder_hard=ponder_hard,
            halt_step=halt_step,
            tokens=x,
            steps_run=executed,
        )


class PlainViT(nn.Module):
    """Unique-block ViT used as a matched-recipe baseline."""

    def __init__(self, cfg: RDVBConfig, depth: int):
        super().__init__()
        self.cfg = cfg
        self.depth = depth
        dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, dim)
        grid = cfg.img_size // cfg.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, dim))
        self.pos_drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, cfg.num_classes)
        self.apply(RDVB._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor, steps: int | None = None, **_ignored) -> ForwardOutput:
        patches = self.patch_embed(images)
        ps = self.cfg.patch_size
        gh, gw = images.shape[-2] // ps, images.shape[-1] // ps
        tokens = torch.cat([self.cls_token.expand(images.shape[0], -1, -1), patches], dim=1)
        tokens = self.pos_drop(tokens + interpolate_pos_embed(self.pos_embed, gh, gw))
        for blk in self.blocks:
            tokens = blk(tokens)
        logits = self.head(self.norm(tokens[:, 0]))
        b = images.shape[0]
        ponder = logits.new_full((b,), float(self.depth))
        halt = torch.full((b, tokens.shape[1]), self.depth, device=images.device, dtype=torch.long)
        return ForwardOutput(logits, [], ponder, ponder, halt, tokens, self.depth)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def block_flops(tokens: int, dim: int, mlp_ratio: float, queries: int | None = None, keys: int | None = None) -> int:
    """Multiply-adds for one block. Q/K/V projections follow the frozen-KV implementation."""
    q_len = tokens if queries is None else queries
    k_len = tokens if keys is None else keys
    qkv = 2 * q_len * dim * dim + 2 * k_len * dim * (2 * dim)
    proj = 2 * q_len * dim * dim
    attn = 4 * q_len * k_len * dim
    mlp = 4 * q_len * dim * int(dim * mlp_ratio)
    return qkv + proj + attn + mlp


def estimate_flops(cfg: RDVBConfig, image_size: int, steps: int, active_frac: float = 1.0) -> int:
    grid = image_size // cfg.patch_size
    tokens = 1 + grid * grid
    dense = block_flops(tokens, cfg.embed_dim, cfg.mlp_ratio)
    queries = max(1, int(round(active_frac * tokens)))
    sparse = block_flops(tokens, cfg.embed_dim, cfg.mlp_ratio, queries=queries, keys=tokens)
    total = dense * (cfg.depth_entry + cfg.depth_exit)
    # First recurrent step is dense if tokens have not halted yet. Later steps use `active_frac`.
    if steps >= 1:
        total += dense
    if steps >= 2:
        total += sparse * (steps - 1)
    return total
