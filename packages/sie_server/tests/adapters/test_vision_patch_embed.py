from __future__ import annotations

import logging

import pytest
import torch
from sie_server.adapters._vision_patch_embed import rebind_vision_patch_embed


def _patch_input(conv: torch.nn.Conv2d | torch.nn.Conv3d, n: int = 5) -> torch.Tensor:
    """A pre-patched input batch shaped (n, C, *kernel) — one row per patch."""
    return torch.randn(n, conv.in_channels, *conv.kernel_size)


class _Conv3dPatchEmbed(torch.nn.Module):
    """Minimal Qwen/GLM-style vision tower: Conv3d patch-embed, kernel == stride, RGB in."""

    def __init__(self) -> None:
        super().__init__()
        self.visual_patch_embed_proj = torch.nn.Conv3d(3, 16, kernel_size=(2, 14, 14), stride=(2, 14, 14))


class _Conv2dPatchEmbed(torch.nn.Module):
    """Minimal PaddleOCR-VL-style vision tower: Conv2d patch-embed, kernel == stride, RGB in."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_embeddings_patch_embedding = torch.nn.Conv2d(3, 16, kernel_size=(14, 14), stride=(14, 14))


class _ConvWithDownsample(torch.nn.Module):
    """Vision tower with both an RGB patch-embed AND a spatial-merge downsample conv.

    Only the RGB-input (in_channels <= 4) conv must be rebound; the downsample
    conv consumes the hidden dim and must be left alone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.visual_patch_embed_proj = torch.nn.Conv3d(3, 16, kernel_size=(2, 14, 14), stride=(2, 14, 14))
        self.visual_downsample = torch.nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(2, 2))


def _assert_rebound_equivalent(model: torch.nn.Module, conv: torch.nn.Conv2d | torch.nn.Conv3d) -> None:
    model.eval()
    probe = _patch_input(conv)
    with torch.inference_mode():
        ref = conv(probe).clone()
    rebind_vision_patch_embed(model, "test")
    with torch.inference_mode():
        got = conv(probe)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_rebinds_conv3d_patch_embed_equivalently() -> None:
    model = _Conv3dPatchEmbed()
    _assert_rebound_equivalent(model, model.visual_patch_embed_proj)


def test_rebinds_conv2d_patch_embed_equivalently() -> None:
    model = _Conv2dPatchEmbed()
    _assert_rebound_equivalent(model, model.vision_embeddings_patch_embedding)


def test_ignores_non_rgb_downsample_conv() -> None:
    model = _ConvWithDownsample()
    down = model.visual_downsample
    probe_down = _patch_input(down)
    with torch.inference_mode():
        down_ref = down(probe_down).clone()
    _assert_rebound_equivalent(model, model.visual_patch_embed_proj)
    with torch.inference_mode():
        down_after = model.visual_downsample(probe_down)
    assert torch.allclose(down_after, down_ref, atol=1e-4, rtol=1e-4)


def test_none_model_is_noop() -> None:
    rebind_vision_patch_embed(None, "test")


def test_no_candidate_conv_skips(caplog: pytest.LogCaptureFixture) -> None:
    class _NoVisionConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

    model = _NoVisionConv()
    with caplog.at_level(logging.WARNING):
        rebind_vision_patch_embed(model, "test")
    assert isinstance(model.linear, torch.nn.Linear)
    assert any("rebind skipped" in r.message for r in caplog.records)


def test_ambiguous_two_rgb_convs_skips() -> None:
    class _TwoPatchEmbeds(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_patch_embed_a = torch.nn.Conv2d(3, 16, kernel_size=14, stride=14)
            self.visual_patch_embed_b = torch.nn.Conv2d(3, 16, kernel_size=14, stride=14)

    model = _TwoPatchEmbeds()
    a, b = model.visual_patch_embed_a, model.visual_patch_embed_b
    a_fwd, b_fwd = a.forward, b.forward
    rebind_vision_patch_embed(model, "test")
    assert a.forward == a_fwd
    assert b.forward == b_fwd
