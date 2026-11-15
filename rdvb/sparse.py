"""Gather/scatter helpers for query-sparse attention.

Halted tokens stay in the key/value sequence. Only live queries are projected
and attended, then scattered back. Padded gather slots (batch rows with fewer
live tokens) are zeroed so they never overwrite a halted token.
"""

from __future__ import annotations

import torch


def active_order(active: torch.Tensor, capacity: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return gather indices, padding mask, and max live count.

    active: bool [B, N]
    take: long [B, M] positions of live tokens, padded with halted tokens
    pad: bool [B, M], True where the slot is padding
    """
    b, n = active.shape
    n_act = active.sum(dim=1)
    # A known capacity is the deployment fast path: no device-to-host scalar
    # read is needed to determine the packed tensor shape.
    m = int(capacity) if capacity is not None else (int(n_act.max().item()) if b > 0 else 0)
    m = min(m, n)
    if m == 0:
        take = active.new_zeros((b, 0), dtype=torch.long)
        pad = active.new_zeros((b, 0), dtype=torch.bool)
        return take, pad, 0
    # Query order does not affect attention: positional information is already
    # in the token state and outputs are scattered back to their source slots.
    # Top-k avoids the O(N log N) full argsort that dominated sparse execution
    # at dense-prediction resolutions.  Boolean 1s are selected before 0s;
    # rows with fewer than M live entries are masked by ``pad`` below.
    take = active.to(dtype=torch.uint8).topk(m, dim=1, sorted=False).indices
    pad = torch.arange(m, device=active.device)[None, :] >= n_act[:, None]
    return take, pad, m


def gather_seq(x: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
    """x [B, N, C], take [B, M] -> [B, M, C]."""
    b, m = take.shape
    return x.gather(1, take.unsqueeze(-1).expand(b, m, x.shape[-1]))


def scatter_seq(src: torch.Tensor, take: torch.Tensor, n: int) -> torch.Tensor:
    """src [B, M, C], take [B, M] -> [B, N, C]."""
    b, m, c = src.shape
    out = src.new_zeros(b, n, c)
    if m == 0:
        return out
    out.scatter_(1, take.unsqueeze(-1).expand(b, m, c), src)
    return out


def ste_mask(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Forward equals `hard`; gradients flow through `soft`."""
    return hard.detach() + soft - soft.detach()
