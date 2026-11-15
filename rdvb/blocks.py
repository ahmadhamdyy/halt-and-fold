"""Pre-norm transformer blocks with query-sparse attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse import active_order, gather_seq, scatter_seq


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def _dense(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
        capacity: int | None = None,
    ) -> torch.Tensor:
        if active is None:
            return self._dense(x)
        n = x.shape[1]
        take, pad, m = active_order(active, capacity)
        if m == 0:
            return torch.zeros_like(x)
        gathered = gather_seq(x, take).masked_fill(pad.unsqueeze(-1), 0)
        out = self._dense(gathered).masked_fill(pad.unsqueeze(-1), 0)
        return scatter_seq(out, take, n)


class Attention(nn.Module):
    """Multi-head self-attention.

    `kv_mode='frozen'` keeps every token in K/V and attends with live queries only.
    `kv_mode='drop'` removes halted tokens from K/V as well.
    Parameter names match DeiT (`qkv`, `proj`) so a pretrained block can be copied.
    """

    def __init__(self, dim: int, num_heads: int, drop: float, kind: str = "softmax"):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if kind not in ("softmax", "linear"):
            raise ValueError(f"unknown attention kind {kind}")
        self.kind = kind
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def _shape(self, t: torch.Tensor, b: int, length: int) -> torch.Tensor:
        return t.view(b, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_pad: torch.Tensor | None) -> torch.Tensor:
        if self.kind == "linear":
            return self._linear_attend(q, k, v, key_pad)
        attn_mask = None
        if key_pad is not None:
            # Added to scores. A padded key is a large negative number, never a bool mask.
            attn_mask = torch.zeros(q.shape[0], 1, q.shape[2], k.shape[2], device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(key_pad[:, None, None, :], torch.finfo(q.dtype).min)
            empty = key_pad.all(dim=1)
            if bool(empty.any()):
                attn_mask = attn_mask.clone()
                attn_mask[empty] = 0
        drop = self.drop if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)

    @staticmethod
    def _feature(x: torch.Tensor) -> torch.Tensor:
        # Positive feature map for normalized kernel attention. Accumulation is
        # performed in float32 below, including under bfloat16 autocast.
        return F.elu(x.float()) + 1.0

    def _linear_attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        qf = self._feature(q)
        kf = self._feature(k)
        vf = v.float()
        if key_pad is not None:
            keep = (~key_pad)[:, None, :, None]
            kf = kf * keep
            vf = vf * keep
        state = kf.transpose(-2, -1) @ vf
        normalizer = kf.sum(dim=2)
        numerator = qf @ state
        denominator = qf @ normalizer.unsqueeze(-1)
        return (numerator / denominator.clamp_min(1e-6)).to(dtype=q.dtype)

    def _merge(self, out: torch.Tensor, b: int, length: int, c: int) -> torch.Tensor:
        return out.transpose(1, 2).reshape(b, length, c)

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
        kv_mode: str = "frozen",
        impl: str = "auto",
        capacity: int | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        gather = active is not None and (impl == "gather" or (impl == "auto" and n >= 1024))
        if not gather:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            # Frozen K/V is dense attention followed by a query mask in the block.
            # Dropping halted tokens needs an explicit key mask.
            key_pad = None
            if active is not None and kv_mode == "drop":
                key_pad = ~active
            out = self._attend(self._shape(q, b, n), self._shape(k, b, n), self._shape(v, b, n), key_pad)
            return self.proj(self._merge(out, b, n, c))

        take, pad, m = active_order(active, capacity)
        if m == 0:
            return torch.zeros_like(x)

        wq, wk, wv = self.qkv.weight.chunk(3, dim=0)
        bq, bk, bv = self.qkv.bias.chunk(3, dim=0)

        if kv_mode == "frozen":
            k = self._shape(F.linear(x, wk, bk), b, n)
            v = self._shape(F.linear(x, wv, bv), b, n)
            gathered = gather_seq(x, take).masked_fill(pad.unsqueeze(-1), 0)
            q = self._shape(F.linear(gathered, wq, bq), b, m)
            out = self._attend(q, k, v, None)
            out = out.masked_fill(pad[:, None, :, None], 0)
            # Project before scatter so the output projection runs only on
            # live queries.
            seq = self.proj(self._merge(out, b, m, c))
            seq = seq.masked_fill(pad.unsqueeze(-1), 0)
            return scatter_seq(seq, take, n)

        if kv_mode != "drop":
            raise ValueError(f"unknown kv_mode {kv_mode}")

        gathered = gather_seq(x, take).masked_fill(pad.unsqueeze(-1), 0)
        qkv = self.qkv(gathered)
        q, k, v = qkv.chunk(3, dim=-1)
        out = self._attend(self._shape(q, b, m), self._shape(k, b, m), self._shape(v, b, m), pad)
        out = out.masked_fill(pad[:, None, :, None], 0)
        seq = self._merge(out, b, m, c).masked_fill(pad.unsqueeze(-1), 0)
        seq = self.proj(seq).masked_fill(pad.unsqueeze(-1), 0)
        return scatter_seq(seq, take, n)

    def forward_cached(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None,
        capacity: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Frozen-K/V attention with exact cross-recurrence projection reuse.

        The recurrent block shares its projection weights at every step. Once
        a token halts, both its state and normalized attention input are
        invariant, so its K/V projections are invariant as well. The first
        call fills a dense cache; later calls replace only live-token entries.
        Active queries still attend to every cached token.
        """
        b, n, c = x.shape
        take, pad, m = active_order(active, capacity)
        if m == 0:
            if cache is None:
                raise ValueError("cannot initialize a K/V cache with no active tokens")
            return torch.zeros_like(x), cache

        gathered = gather_seq(x, take).masked_fill(pad.unsqueeze(-1), 0)
        q_seq, k_live, v_live = self.qkv(gathered).chunk(3, dim=-1)

        if cache is None:
            # The first call is fully active, so the fused projections above
            # also initialize the complete cache.
            k_cache = k_live
            v_cache = v_live
        else:
            k_cache, v_cache, previous_active, previous_capacity = cache
            live_index = take.unsqueeze(-1).expand(b, m, c)
            old_k = gather_seq(k_cache, take)
            old_v = gather_seq(v_cache, take)
            k_live = torch.where(pad.unsqueeze(-1), old_k, k_live)
            v_live = torch.where(pad.unsqueeze(-1), old_v, v_live)
            # Cache execution is inference-only, so update the persistent
            # buffers in place rather than copying the full [B,N,C] cache.
            k_cache.scatter_(1, live_index, k_live)
            v_cache.scatter_(1, live_index, v_live)

            # A token that just halted changed during its last active update.
            # Write its final K/V once. Concatenating Wk and Wv makes this one
            # projection launch; continuing tokens use fused QKV above.
            newly_halted = previous_active & ~active
            halted_capacity = (
                max(previous_capacity - m, 0) if capacity is not None else None
            )
            halt_take, halt_pad, halt_m = active_order(newly_halted, halted_capacity)
            if halt_m:
                halted_x = gather_seq(x, halt_take).masked_fill(halt_pad.unsqueeze(-1), 0)
                _, wk, wv = self.qkv.weight.chunk(3, dim=0)
                _, bk, bv = self.qkv.bias.chunk(3, dim=0)
                kv_new = F.linear(halted_x, torch.cat([wk, wv]), torch.cat([bk, bv]))
                k_new, v_new = kv_new.chunk(2, dim=-1)
                halt_index = halt_take.unsqueeze(-1).expand(b, halt_m, c)
                old_k = gather_seq(k_cache, halt_take)
                old_v = gather_seq(v_cache, halt_take)
                k_new = torch.where(halt_pad.unsqueeze(-1), old_k, k_new)
                v_new = torch.where(halt_pad.unsqueeze(-1), old_v, v_new)
                k_cache.scatter_(1, halt_index, k_new)
                v_cache.scatter_(1, halt_index, v_new)

        q = self._shape(q_seq, b, m)
        k = self._shape(k_cache, b, n)
        v = self._shape(v_cache, b, n)
        out = self._attend(q, k, v, None)
        out = out.masked_fill(pad[:, None, :, None], 0)
        seq = self.proj(self._merge(out, b, m, c)).masked_fill(pad.unsqueeze(-1), 0)
        return scatter_seq(seq, take, n), (k_cache, v_cache, active, m)

    def forward_folded(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None,
        capacity: int | None = None,
        writeback_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Exact halt-and-fold execution for normalized linear attention.

        Halted tokens are written once into per-head sufficient statistics.
        Live-token statistics are recomputed each recurrence and combined with
        this persistent memory, exactly matching full linear attention.
        """
        if self.kind != "linear":
            raise ValueError("halt-and-fold requires linear attention")
        b, n, c = x.shape
        take, pad, m = active_order(active, capacity)
        if m == 0:
            if memory is None:
                raise ValueError("cannot initialize folded memory with no active tokens")
            return torch.zeros_like(x), memory

        live_x = gather_seq(x, take).masked_fill(pad.unsqueeze(-1), 0)
        q_seq, k_seq, v_seq = self.qkv(live_x).chunk(3, dim=-1)
        q = self._shape(q_seq, b, m)
        k = self._shape(k_seq, b, m)
        v = self._shape(v_seq, b, m)

        d = self.head_dim
        if memory is None:
            halted_state = x.new_zeros((b, self.num_heads, d, d), dtype=torch.float32)
            halted_norm = x.new_zeros((b, self.num_heads, d), dtype=torch.float32)
        else:
            halted_state, halted_norm, previous_active, previous_capacity = memory
            newly_halted = previous_active & ~active
            halted_capacity = max(previous_capacity - m, 0) if capacity is not None else None
            halt_take, halt_pad, halt_m = active_order(newly_halted, halted_capacity)
            if halt_m:
                source = x if writeback_source is None else writeback_source
                halted_x = gather_seq(source, halt_take).masked_fill(halt_pad.unsqueeze(-1), 0)
                _, wk, wv = self.qkv.weight.chunk(3, dim=0)
                _, bk, bv = self.qkv.bias.chunk(3, dim=0)
                kv = F.linear(halted_x, torch.cat([wk, wv]), torch.cat([bk, bv]))
                kh, vh = kv.chunk(2, dim=-1)
                kh = self._feature(self._shape(kh, b, halt_m))
                vh = self._shape(vh, b, halt_m).float()
                keep = (~halt_pad)[:, None, :, None]
                kh = kh * keep
                vh = vh * keep
                halted_state = halted_state + kh.transpose(-2, -1) @ vh
                halted_norm = halted_norm + kh.sum(dim=2)

        kf = self._feature(k)
        vf = v.float()
        keep = (~pad)[:, None, :, None]
        kf = kf * keep
        vf = vf * keep
        state = halted_state + kf.transpose(-2, -1) @ vf
        normalizer = halted_norm + kf.sum(dim=2)
        qf = self._feature(q)
        numerator = qf @ state
        denominator = qf @ normalizer.unsqueeze(-1)
        out = (numerator / denominator.clamp_min(1e-6)).to(dtype=x.dtype)
        out = out.masked_fill(pad[:, None, :, None], 0)
        seq = self.proj(self._merge(out, b, m, c)).masked_fill(pad.unsqueeze(-1), 0)
        return scatter_seq(seq, take, n), (halted_state, halted_norm, active, m)


class ViTBlock(nn.Module):
    """DeiT-style pre-norm block. Live-token outputs are masked before the residual add."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attention_kind: str = "softmax",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, drop, kind=attention_kind)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
        kv_mode: str = "frozen",
        impl: str = "auto",
        modulation: tuple[torch.Tensor, ...] | None = None,
        capacity: int | None = None,
    ) -> torch.Tensor:
        """Apply a pre-norm block, optionally modulated by recurrent depth.

        ``modulation`` contains two FiLM pairs and two residual gates:
        ``(scale_attn, shift_attn, scale_mlp, shift_mlp, gate_attn,
        gate_mlp)``.  Every tensor has shape ``[C]``.  Zero FiLM values and
        unit gates recover the ordinary DeiT block exactly, which makes this
        usable with converted checkpoints.
        """
        gather = active is not None and (impl == "gather" or (impl == "auto" and x.shape[1] >= 1024))
        mlp_active = active if gather else None
        if modulation is None:
            scale_attn = shift_attn = scale_mlp = shift_mlp = None
            gate_attn = gate_mlp = None
        else:
            scale_attn, shift_attn, scale_mlp, shift_mlp, gate_attn, gate_mlp = modulation
        # Pre-norm: the MLP sees the attention residual, matching DeiT.
        normed = self.norm1(x)
        if scale_attn is not None:
            normed = normed * (1.0 + scale_attn.view(1, 1, -1)) + shift_attn.view(1, 1, -1)
        h = self.attn(normed, active=active, kv_mode=kv_mode, impl=impl, capacity=capacity)
        if gate_attn is not None:
            h = h * gate_attn.view(1, 1, -1)
        if active is not None:
            h = h * active.unsqueeze(-1).to(dtype=x.dtype)
        x = x + h
        normed = self.norm2(x)
        if scale_mlp is not None:
            normed = normed * (1.0 + scale_mlp.view(1, 1, -1)) + shift_mlp.view(1, 1, -1)
        m = self.mlp(normed, active=mlp_active, capacity=capacity)
        if gate_mlp is not None:
            m = m * gate_mlp.view(1, 1, -1)
        if active is not None:
            m = m * active.unsqueeze(-1).to(dtype=x.dtype)
        return x + m

    def forward_cached(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None,
        capacity: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Apply the block while reusing invariant recurrent K/V projections."""
        normed = self.norm1(x)
        h, cache = self.attn.forward_cached(normed, active, cache, capacity)
        h = h * active.unsqueeze(-1).to(dtype=x.dtype)
        x = x + h
        m = self.mlp(self.norm2(x), active=active, capacity=capacity)
        m = m * active.unsqueeze(-1).to(dtype=x.dtype)
        return x + m, cache

    def forward_folded(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None,
        capacity: int | None = None,
        writeback_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        normed = self.norm1(x)
        h, memory = self.attn.forward_folded(
            normed, active, memory, capacity, writeback_source=writeback_source
        )
        h = h * active.unsqueeze(-1).to(dtype=x.dtype)
        x = x + h
        m = self.mlp(self.norm2(x), active=active, capacity=capacity)
        m = m * active.unsqueeze(-1).to(dtype=x.dtype)
        return x + m, memory
