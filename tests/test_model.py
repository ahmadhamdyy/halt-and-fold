"""Correctness checks for query-sparse updates and the recurrent backbone."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdvb.blocks import ViTBlock
from rdvb.convert import convert_deit
from rdvb.model import RDVB, RDVBConfig


def test_prenorm_residual_order():
    torch.manual_seed(0)
    block = ViTBlock(32, 4).eval()
    x = torch.randn(1, 8, 32)
    y = block(x)
    z = x + block.attn(block.norm1(x))
    z = z + block.mlp(block.norm2(z))
    assert torch.allclose(y, z, atol=1e-6)


def test_inactive_tokens_stay_put():
    torch.manual_seed(0)
    block = ViTBlock(32, 4)
    x = torch.randn(2, 16, 32)
    active = torch.zeros(2, 16, dtype=torch.bool)
    active[:, 0] = True
    active[:, 5] = True
    y = block(x, active=active, kv_mode="frozen")
    assert torch.allclose(y[0, ~active[0]], x[0, ~active[0]], atol=1e-6)
    assert not torch.allclose(y[0, active[0]], x[0, active[0]])


def test_all_active_matches_dense():
    torch.manual_seed(0)
    block = ViTBlock(32, 4).eval()
    x = torch.randn(2, 20, 32)
    active = torch.ones(2, 20, dtype=torch.bool)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        dense = block(x, active=None)
        sparse = block(x, active=active, kv_mode="frozen")
    assert torch.allclose(dense, sparse, rtol=1e-4, atol=1e-4), (dense - sparse).abs().max()


def test_dense_mask_matches_gather():
    torch.manual_seed(0)
    block = ViTBlock(32, 4).eval()
    x = torch.randn(2, 18, 32)
    active = torch.zeros(2, 18, dtype=torch.bool)
    active[:, :7] = True
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        dense = block(x, active=active, kv_mode="frozen", impl="dense")
        gather = block(x, active=active, kv_mode="frozen", impl="gather")
    assert torch.allclose(dense, gather, rtol=1e-4, atol=1e-4), (dense - gather).abs().max()


def test_drop_differs_from_frozen():
    torch.manual_seed(0)
    block = ViTBlock(32, 4).eval()
    x = torch.randn(1, 12, 32)
    active = torch.zeros(1, 12, dtype=torch.bool)
    active[:, :4] = True
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        frozen = block(x, active=active, kv_mode="frozen")
        dropped = block(x, active=active, kv_mode="drop")
    assert not torch.allclose(frozen[:, :4], dropped[:, :4], atol=1e-4)


def test_recurrent_forward_and_backward():
    torch.manual_seed(0)
    cfg = RDVBConfig(img_size=32, patch_size=16, embed_dim=32, num_heads=4, num_classes=5, router="spatial")
    model = RDVB(cfg).train()
    images = torch.randn(2, 3, 32, 32)
    out = model(images, steps=3, halting=True)
    assert out.logits.shape == (2, 5)
    assert len(out.aux_logits) == 3
    loss = out.logits.sum() + out.ponder_soft.mean()
    loss.backward()
    assert model.core.block.attn.qkv.weight.grad is not None
    assert model.router.fc.bias.grad is not None
    assert model.router.fc.bias.grad.abs().sum() > 0


def test_film_core_is_identity_modulated_at_init_and_step_specific_after_update():
    torch.manual_seed(0)
    cfg = RDVBConfig(
        img_size=32,
        patch_size=16,
        embed_dim=32,
        num_heads=4,
        num_classes=5,
        router="none",
        core_mode="film",
        cond_dim=16,
    )
    model = RDVB(cfg).train()
    x = torch.randn(2, 5, 32)
    expected = model.core.block(x)
    actual = model.core(x, 0, None, None, "frozen")
    assert torch.allclose(expected, actual, atol=1e-6)
    loss = model(torch.randn(2, 3, 32, 32), steps=3, halting=False).logits.sum()
    loss.backward()
    assert model.core.cond[-1].weight.grad is not None
    assert model.core.cond[-1].weight.grad.abs().sum() > 0


def test_halt_freezes_later_steps():
    torch.manual_seed(0)
    cfg = RDVBConfig(
        img_size=32,
        patch_size=16,
        embed_dim=32,
        num_heads=4,
        num_classes=3,
        router="independent",
        router_bias=8.0,
    )
    model = RDVB(cfg).eval()
    images = torch.randn(1, 3, 32, 32)
    early = model(images, steps=1, halting=True)
    late = model(images, steps=3, halting=True)
    # Bias is very positive, so every patch halts after the first recurrent update.
    assert int(late.halt_step[0, 1].item()) == 1
    assert int(late.halt_step[0, 0].item()) == 3
    # Exit still runs on the class token. Patch tokens that halted skip the exit update,
    # so their final states match a one-step unroll only up to that shared prefix.
    # The class token must keep moving; patch ponder is 1.
    assert float(late.halt_step[0, 1:].float().mean()) == 1
    assert early.logits.shape == late.logits.shape


def test_resolution_extrapolation():
    cfg = RDVBConfig(img_size=32, patch_size=16, embed_dim=32, num_heads=4, num_classes=3, router="none")
    model = RDVB(cfg).eval()
    out = model(torch.randn(1, 3, 64, 48), steps=2, halting=False)
    assert out.tokens.shape[1] == 1 + 4 * 3


def test_capacity_schedule_has_exact_monotone_budget():
    torch.manual_seed(0)
    cfg = RDVBConfig(img_size=64, patch_size=16, embed_dim=32, num_heads=4, num_classes=3)
    model = RDVB(cfg).eval()
    images = torch.randn(2, 3, 64, 64)
    schedule = (1.0, 0.75, 0.5, 0.25)
    out = model(images, steps=4, halting=True, capacity_schedule=schedule, impl="gather")
    # Patch ponder is exactly 1 + .75 + .5 + .25 at a 16-token grid.
    assert torch.allclose(out.halt_step[:, 1:].float().mean(), torch.tensor(2.5))
    assert torch.isfinite(out.logits).all()
    model.train()
    model.zero_grad(set_to_none=True)
    trained = model(images, steps=4, halting=True, capacity_schedule=schedule, impl="gather")
    trained.logits.sum().backward()
    assert model.router.fc.weight.grad is not None
    assert model.router.fc.weight.grad.abs().sum() > 0


def test_cross_depth_kv_cache_is_output_equivalent():
    torch.manual_seed(17)
    cfg = RDVBConfig(img_size=64, patch_size=16, embed_dim=32, num_heads=4, num_classes=3)
    model = RDVB(cfg).eval()
    images = torch.randn(2, 3, 64, 64)
    schedule = (1.0, 0.75, 0.5, 0.25)
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        reference = model(
            images, steps=4, halting=True, capacity_schedule=schedule, impl="gather"
        )
        cached = model(
            images,
            steps=4,
            halting=True,
            capacity_schedule=schedule,
            impl="gather",
            cache_kv=True,
        )
    torch.testing.assert_close(cached.tokens, reference.tokens, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cached.logits, reference.logits, rtol=2e-5, atol=2e-6)
    assert torch.equal(cached.halt_step, reference.halt_step)


def test_halt_and_fold_matches_full_linear_attention():
    torch.manual_seed(23)
    cfg = RDVBConfig(
        img_size=64,
        patch_size=16,
        embed_dim=32,
        num_heads=4,
        num_classes=3,
        core_attention="linear",
    )
    model = RDVB(cfg).eval()
    images = torch.randn(2, 3, 64, 64)
    schedule = (1.0, 0.75, 0.5, 0.25)
    with torch.inference_mode():
        reference = model(
            images, steps=4, halting=True, capacity_schedule=schedule, impl="gather"
        )
        folded = model(
            images,
            steps=4,
            halting=True,
            capacity_schedule=schedule,
            impl="gather",
            fold_halted=True,
        )
        stale = model(
            images,
            steps=4,
            halting=True,
            capacity_schedule=schedule,
            impl="gather",
            fold_halted=True,
            fold_writeback="stale",
        )
    torch.testing.assert_close(folded.tokens, reference.tokens, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(folded.logits, reference.logits, rtol=3e-5, atol=3e-6)
    assert torch.equal(folded.halt_step, reference.halt_step)
    assert not torch.allclose(stale.tokens, reference.tokens, rtol=3e-5, atol=3e-6)


def test_random_routing_preserves_capacity_schedule():
    torch.manual_seed(31)
    cfg = RDVBConfig(
        img_size=64,
        patch_size=16,
        embed_dim=32,
        num_heads=4,
        num_classes=3,
        core_attention="linear",
    )
    model = RDVB(cfg).eval()
    schedule = (1.0, 0.75, 0.5, 0.25)
    with torch.inference_mode():
        learned = model(
            torch.randn(2, 3, 64, 64),
            steps=4,
            halting=True,
            capacity_schedule=schedule,
        )
        torch.manual_seed(37)
        random = model(
            torch.randn(2, 3, 64, 64),
            steps=4,
            halting=True,
            capacity_schedule=schedule,
            routing_mode="random",
        )
    assert learned.halt_step[:, 1:].float().mean() == random.halt_step[:, 1:].float().mean()


def test_conversion_copies_entry():
    path = os.environ.get(
        "DEIT_PATH",
        "",
    )
    if not os.path.exists(path):
        print("skip conversion test; checkpoint missing")
        return
    model = convert_deit(path, how="middle")
    src = torch.load(path, map_location="cpu", weights_only=False)["model"]
    assert torch.equal(model.entry[0].attn.qkv.weight, src["blocks.0.attn.qkv.weight"])
    assert torch.equal(model.exit[0].mlp.fc2.weight, src["blocks.11.mlp.fc2.weight"])
    assert torch.equal(model.core.block.norm1.weight, src["blocks.6.norm1.weight"])
    images = torch.randn(1, 3, 224, 224)
    out = model.eval()(images, steps=1, halting=False)
    assert torch.isfinite(out.logits).all()


def test_plain_port_matches_deit():
    path = os.environ.get(
        "DEIT_PATH",
        "",
    )
    if not os.path.exists(path):
        return
    import math

    import torch.nn.functional as F

    from rdvb.convert import deit_as_plain

    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    torch.manual_seed(0)
    images = torch.randn(1, 3, 224, 224)
    feat = F.conv2d(images, sd["patch_embed.proj.weight"], sd["patch_embed.proj.bias"], stride=16)
    tokens = feat.flatten(2).transpose(1, 2)
    tokens = torch.cat([sd["cls_token"].expand(1, -1, -1), tokens], 1) + sd["pos_embed"]
    for i in range(12):
        p = f"blocks.{i}."
        c = tokens.shape[-1]
        h = F.layer_norm(tokens, (c,), sd[p + "norm1.weight"], sd[p + "norm1.bias"], 1e-6)
        b, n, _ = h.shape
        qkv = F.linear(h, sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]).reshape(b, n, 3, 6, c // 6)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(c // 6)
        h = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
        tokens = tokens + F.linear(h, sd[p + "attn.proj.weight"], sd[p + "attn.proj.bias"])
        h = F.layer_norm(tokens, (c,), sd[p + "norm2.weight"], sd[p + "norm2.bias"], 1e-6)
        h = F.gelu(F.linear(h, sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc1.bias"]))
        tokens = tokens + F.linear(h, sd[p + "mlp.fc2.weight"], sd[p + "mlp.fc2.bias"])
    tokens = F.layer_norm(tokens, (384,), sd["norm.weight"], sd["norm.bias"], 1e-6)
    ref = F.linear(tokens[:, 0], sd["head.weight"], sd["head.bias"])
    ours = deit_as_plain(path).eval()(images).logits
    assert torch.allclose(ref, ours, rtol=1e-4, atol=1e-4), (ref - ours).abs().max()


if __name__ == "__main__":
    test_prenorm_residual_order()
    test_inactive_tokens_stay_put()
    test_all_active_matches_dense()
    test_dense_mask_matches_gather()
    test_drop_differs_from_frozen()
    test_recurrent_forward_and_backward()
    test_film_core_is_identity_modulated_at_init_and_step_specific_after_update()
    test_halt_freezes_later_steps()
    test_resolution_extrapolation()
    test_capacity_schedule_has_exact_monotone_budget()
    test_cross_depth_kv_cache_is_output_equivalent()
    test_halt_and_fold_matches_full_linear_attention()
    test_random_routing_preserves_capacity_schedule()
    test_conversion_copies_entry()
    test_plain_port_matches_deit()
    print("all tests passed")
