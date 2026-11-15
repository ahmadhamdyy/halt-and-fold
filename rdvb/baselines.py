"""Dynamic-compute and plain vision backbones in a shared training format.

These compact implementations share the ImageNette training recipe with the
recurrent model. They are not full reproductions of the original methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ViTBlock
from .model import PRESETS, ForwardOutput, PatchEmbed, RDVB, RDVBConfig, interpolate_pos_embed
from .sparse import ste_mask


def _result(logits, aux, ponder, depth, tokens) -> ForwardOutput:
    b = logits.shape[0]
    ponder_b = ponder if ponder.ndim == 1 else ponder.expand(b)
    halt = torch.full((b, tokens.shape[1]), int(depth), device=logits.device, dtype=torch.long)
    return ForwardOutput(logits, aux, ponder_b, ponder_b, halt, tokens, int(depth))


def _stem(cfg: RDVBConfig):
    dim = cfg.embed_dim
    patch = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, dim)
    grid = cfg.img_size // cfg.patch_size
    cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, dim))
    return patch, cls_token, pos_embed


def _embed(patch, cls_token, pos_embed, drop, images):
    tokens = patch(images)
    ps = patch.patch_size
    gh, gw = images.shape[-2] // ps, images.shape[-1] // ps
    tokens = torch.cat([cls_token.expand(images.shape[0], -1, -1), tokens], dim=1)
    return drop(tokens + interpolate_pos_embed(pos_embed, gh, gw))


def _init_vit(module: nn.Module) -> None:
    module.apply(RDVB._init_weights)
    if hasattr(module, "pos_embed"):
        nn.init.trunc_normal_(module.pos_embed, std=0.02)
    if hasattr(module, "cls_token"):
        nn.init.trunc_normal_(module.cls_token, std=0.02)


class EarlyExitViT(nn.Module):
    """Unique-block ViT with a classifier after every block.

    Training supervises every exit. At test time a sample stops at the first
    exit whose confidence is above `exit_threshold`.
    """

    def __init__(self, cfg: RDVBConfig, depth: int = 12, exit_threshold: float = 0.7):
        super().__init__()
        self.cfg = cfg
        self.depth = depth
        self.exit_threshold = exit_threshold
        self.patch_embed, self.cls_token, self.pos_embed = _stem(cfg)
        self.pos_drop = nn.Dropout(cfg.drop)
        dim = cfg.embed_dim
        self.blocks = nn.ModuleList(ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-6) for _ in range(depth))
        self.heads = nn.ModuleList(nn.Linear(dim, cfg.num_classes) for _ in range(depth))
        _init_vit(self)

    def forward(self, images, steps: int | None = None, **_ignored) -> ForwardOutput:
        x = _embed(self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, images)
        aux = []
        for block, norm, head in zip(self.blocks, self.norms, self.heads):
            x = block(x)
            aux.append(head(norm(x[:, 0])))
        if self.training or self.exit_threshold >= 1:
            chosen = aux[-1]
            depth = x.new_full((x.shape[0],), float(self.depth))
        else:
            chosen = aux[-1]
            exited = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            depth = x.new_full((x.shape[0],), float(self.depth))
            for index, logits in enumerate(aux):
                confident = torch.softmax(logits, dim=-1).max(dim=-1).values >= self.exit_threshold
                take = confident & ~exited
                chosen = torch.where(take.unsqueeze(-1), logits, chosen)
                depth = torch.where(take, depth.new_full(depth.shape, float(index + 1)), depth)
                exited = exited | take
        return _result(chosen, aux, depth, self.depth, x)


class TokenPruneViT(nn.Module):
    """DynamicViT / EViT style: a learned score drops patches at two depths.

    The class token is always kept. Dropped patches leave the sequence, so
    later blocks cannot attend to them.
    """

    def __init__(self, cfg: RDVBConfig, depth: int = 12, keep_ratio: float = 0.5, prune_at: tuple[int, ...] = (4, 8)):
        super().__init__()
        self.cfg = cfg
        self.depth = depth
        self.keep_ratio = keep_ratio
        self.prune_at = set(prune_at)
        self.patch_embed, self.cls_token, self.pos_embed = _stem(cfg)
        self.pos_drop = nn.Dropout(cfg.drop)
        dim = cfg.embed_dim
        self.blocks = nn.ModuleList(ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(depth))
        self.scorers = nn.ModuleDict({str(i): nn.Linear(dim, 1) for i in prune_at})
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, cfg.num_classes)
        _init_vit(self)

    def _prune(self, tokens: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        raw = scorer(tokens).squeeze(-1)
        score = torch.cat([raw[:, :1] + 1e4, raw[:, 1:]], dim=1)
        n = tokens.shape[1]
        n_keep = max(2, int(round(self.keep_ratio * (n - 1))) + 1)
        n_keep = min(n_keep, n)
        index = score.topk(n_keep, dim=1).indices.sort(dim=1).values
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep.scatter_(1, index, True)
        soft = torch.sigmoid(score / 10.0)
        gate = ste_mask(keep.float(), soft).unsqueeze(-1)
        gathered = (tokens * gate).gather(1, index.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return gathered

    def forward(self, images, steps: int | None = None, **_ignored) -> ForwardOutput:
        x = _embed(self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, images)
        lengths = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in self.prune_at:
                x = self._prune(x, self.scorers[str(index)])
            lengths.append(x.shape[1])
        logits = self.head(self.norm(x[:, 0]))
        ponder = x.new_full((x.shape[0],), float(sum(lengths) / len(lengths)))
        return _result(logits, [], ponder, self.depth, x)


class ToMeViT(nn.Module):
    """Token Merging: bipartite soft matching shortens the sequence every block.

    Matching follows the published ToMe rule. The class token is never merged.
    """

    def __init__(self, cfg: RDVBConfig, depth: int = 12, merge_r: int = 8):
        super().__init__()
        self.cfg = cfg
        self.depth = depth
        self.merge_r = merge_r
        self.patch_embed, self.cls_token, self.pos_embed = _stem(cfg)
        self.pos_drop = nn.Dropout(cfg.drop)
        dim = cfg.embed_dim
        self.blocks = nn.ModuleList(ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, cfg.num_classes)
        _init_vit(self)

    def _merge(self, tokens: torch.Tensor) -> torch.Tensor:
        cls, patches = tokens[:, :1], tokens[:, 1:]
        if patches.shape[1] < 4 or self.merge_r <= 0:
            return tokens
        metric = F.normalize(patches, dim=-1)
        a, b = metric[:, ::2], metric[:, 1::2]
        xa, xb = patches[:, ::2], patches[:, 1::2]
        r = min(self.merge_r, a.shape[1])
        scores = a @ b.transpose(-1, -2)
        best_val, best_dst = scores.max(dim=-1)
        edge = best_val.topk(r, dim=-1).indices
        keep = torch.ones(a.shape[0], a.shape[1], dtype=torch.bool, device=tokens.device)
        keep.scatter_(1, edge, False)
        n_keep = a.shape[1] - r
        order = keep.float().argsort(dim=1, descending=True)[:, :n_keep]
        xa_keep = xa.gather(1, order.unsqueeze(-1).expand(-1, -1, xa.shape[-1]))
        src = xa.gather(1, edge.unsqueeze(-1).expand(-1, -1, xa.shape[-1]))
        dst = best_dst.gather(1, edge)
        added = torch.zeros_like(xb)
        added.scatter_add_(1, dst.unsqueeze(-1).expand_as(src), src)
        counts = torch.ones(xb.shape[0], xb.shape[1], 1, device=tokens.device, dtype=tokens.dtype)
        counts = counts.scatter_add(1, dst.unsqueeze(-1), torch.ones_like(src[:, :, :1]))
        xb_new = (xb + added) / counts
        return torch.cat([cls, xa_keep, xb_new], dim=1)

    def forward(self, images, steps: int | None = None, **_ignored) -> ForwardOutput:
        x = _embed(self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, images)
        lengths = []
        for block in self.blocks:
            x = block(x)
            x = self._merge(x)
            lengths.append(x.shape[1])
        logits = self.head(self.norm(x[:, 0]))
        ponder = x.new_full((x.shape[0],), float(sum(lengths) / len(lengths)))
        return _result(logits, [], ponder, self.depth, x)


class MixtureOfDepthsViT(nn.Module):
    """Mixture-of-Depths routing: each block updates only the top-capacity tokens.

    Skipped tokens take the residual path and stay in the sequence.
    """

    def __init__(self, cfg: RDVBConfig, depth: int = 12, capacity: float = 0.5):
        super().__init__()
        self.cfg = cfg
        self.depth = depth
        self.capacity = capacity
        self.patch_embed, self.cls_token, self.pos_embed = _stem(cfg)
        self.pos_drop = nn.Dropout(cfg.drop)
        dim = cfg.embed_dim
        self.blocks = nn.ModuleList(ViTBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop) for _ in range(depth))
        self.routers = nn.ModuleList(nn.Linear(dim, 1) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, cfg.num_classes)
        _init_vit(self)
        for router in self.routers:
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)

    def forward(self, images, steps: int | None = None, **_ignored) -> ForwardOutput:
        x = _embed(self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, images)
        used = []
        for block, router in zip(self.blocks, self.routers):
            score = router(x).squeeze(-1)
            score = torch.cat([score.new_full((score.shape[0], 1), 1e4), score[:, 1:]], dim=1)
            n_keep = max(2, int(round(self.capacity * (x.shape[1] - 1))) + 1)
            index = score.topk(n_keep, dim=1).indices
            hard = torch.zeros_like(score, dtype=torch.bool)
            hard.scatter_(1, index, True)
            soft = torch.sigmoid(score)
            gate = ste_mask(hard.float(), soft)
            y = block(x)
            x = x + gate.unsqueeze(-1) * (y - x)
            used.append(gate.mean(dim=1))
        logits = self.head(self.norm(x[:, 0]))
        ponder = torch.stack(used, dim=0).mean(dim=0) * self.depth
        return _result(logits, [], ponder, self.depth, x)


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window * window, c)


def window_reverse(windows: torch.Tensor, window: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window / window))
    x = windows.view(b, h // window, w // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window - 1) * (2 * window - 1), num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        coords = coords.flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window - 1
        relative[:, :, 1] += window - 1
        relative[:, :, 0] *= 2 * window - 1
        index = relative.sum(-1).view(-1)
        self.register_buffer("relative_position_index", index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nwin = mask.shape[0]
            attn = attn.view(b // nwin, nwin, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int, shift: int):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = WindowAttention(dim, num_heads, window)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, h: int, w: int, mask: torch.Tensor | None) -> torch.Tensor:
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(x, self.window)
        windows = self.attn(windows, mask if self.shift else None)
        x = window_reverse(windows, self.window, h, w)
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(b, h * w, c)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim, eps=1e-6)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(b, -1, 4 * c)
        return self.reduction(self.norm(x)), h // 2, w // 2


class SwinTiny(nn.Module):
    """Swin-T shifted-window hierarchical baseline."""

    def __init__(self, num_classes: int, img_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.depths = (2, 2, 6, 2)
        self.dims = (96, 192, 384, 768)
        self.heads = (3, 6, 12, 24)
        self.window = 7
        self.patch_embed = nn.Conv2d(3, self.dims[0], kernel_size=4, stride=4)
        self.pos_drop = nn.Dropout(0.0)
        stages = []
        merges = []
        for stage, (depth, dim, heads) in enumerate(zip(self.depths, self.dims, self.heads)):
            blocks = []
            for i in range(depth):
                shift = 0 if i % 2 == 0 else self.window // 2
                blocks.append(SwinBlock(dim, heads, self.window, shift))
            stages.append(nn.ModuleList(blocks))
            if stage < 3:
                merges.append(PatchMerging(dim))
        self.stages = nn.ModuleList(stages)
        self.merges = nn.ModuleList(merges)
        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes)
        self.apply(RDVB._init_weights)
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _attn_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device.type, device.index)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        img = torch.zeros(1, h, w, 1, device=device)
        slices = (slice(0, -self.window), slice(-self.window, -self.window // 2), slice(-self.window // 2, None))
        count = 0
        for top in slices:
            for left in slices:
                img[:, top, left, :] = count
                count += 1
        windows = window_partition(img, self.window).squeeze(-1)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, torch.finfo(torch.float32).min).masked_fill(mask == 0, 0.0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, images, steps: int | None = None, **_ignored) -> ForwardOutput:
        x = self.patch_embed(images)
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        h, w = images.shape[-2] // 4, images.shape[-1] // 4
        for stage_index, blocks in enumerate(self.stages):
            mask = self._attn_mask(h, w, x.device)
            for block in blocks:
                x = block(x, h, w, mask)
            if stage_index < len(self.merges):
                x, h, w = self.merges[stage_index](x, h, w)
        logits = self.head(self.norm(x).mean(dim=1))
        ponder = logits.new_full((logits.shape[0],), float(sum(self.depths)))
        return _result(logits, [], ponder, sum(self.depths), x)


def build_baseline(name: str, num_classes: int, preset: str = "s"):
    spec = PRESETS[preset]
    cfg = RDVBConfig(num_classes=num_classes, embed_dim=spec["embed_dim"], num_heads=spec["num_heads"], router="none")
    if name == "plain3":
        from .model import PlainViT

        return PlainViT(cfg, depth=3)
    if name == "plain6":
        from .model import PlainViT

        return PlainViT(cfg, depth=6)
    if name == "plain12":
        from .model import PlainViT

        return PlainViT(cfg, depth=12)
    if name == "early_exit":
        return EarlyExitViT(cfg, depth=12)
    if name == "token_prune":
        return TokenPruneViT(cfg, depth=12)
    if name == "tome":
        return ToMeViT(cfg, depth=12, merge_r=8)
    if name == "mod":
        return MixtureOfDepthsViT(cfg, depth=12, capacity=0.5)
    if name == "swin_t":
        return SwinTiny(num_classes)
    raise ValueError(name)
