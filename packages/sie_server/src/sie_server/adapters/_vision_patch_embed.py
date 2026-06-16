from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rebind_vision_patch_embed(model: torch.nn.Module | None, label: str) -> None:
    """Rebind a VLM vision patch-embed Conv to an equivalent ``F.linear``.

    Several VLM vision encoders embed patches with a non-overlapping conv
    (``kernel == stride``) applied to *pre-patched* input ``(num_patches, …)``.
    On CUDA cuDNN dispatches that as ~one tiny kernel launch per patch (tens of
    thousands per page), so the GPU sits idle while a single CPU thread launches
    kernels — the dominant cost of every request. The op is mathematically a
    per-patch linear projection, so rebinding ``conv.forward`` to a single
    ``F.linear`` GEMM (same weights) is numerically identical and orders of
    magnitude faster.

    Discovers the patch-embed by structure (the unique vision conv with
    ``kernel == stride`` and RGB input, ``in_channels <= 4``) rather than a
    hardcoded attribute path, so it handles every affected backbone (Qwen2/2.5/3-VL
    and GLM-4V Conv3d, PaddleOCR-VL Conv2d). Self-validating: a small synthetic
    patch batch is run through both the original conv and the matmul at load time,
    and the fast path is kept only if the outputs match (within bf16 rounding);
    otherwise a warning is logged and the original conv is left in place. No
    weights are changed and the check stays off the request path.

    Best-effort and idempotent in effect: on any structural mismatch or numeric
    divergence it logs and returns, leaving the original conv untouched.

    Args:
        model: the loaded adapter model (the full ``nn.Module`` tree is scanned).
        label: short adapter tag used only as a log prefix, e.g. ``"glm_ocr"``.
    """
    if model is None:
        return

    # Non-overlapping convs (kernel == stride) inside the vision tower. The patch
    # embed is one such conv, but the tower may have others — e.g. a spatial-merge
    # ``downsample`` conv (also kernel == stride). The patch embed is the
    # pathological one (~one tiny kernel launch per patch); disambiguate by input
    # channels: the patch embed consumes raw image channels (RGB, <= 4), while
    # merge/downsample convs consume the hidden dim.
    convs: list[tuple[str, torch.nn.Conv2d | torch.nn.Conv3d]] = []
    for name, mod in model.named_modules():
        lname = name.lower()
        if (
            isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv3d))
            and tuple(mod.kernel_size) == tuple(mod.stride)
            and ("vis" in lname or "patch" in lname or "embed" in lname)
        ):
            convs.append((name, mod))

    # Identify the patch embed strictly by structure: it is the only vision conv
    # that consumes raw image channels (RGB). Any other kernel==stride conv (e.g.
    # a spatial-merge downsample) consumes the hidden dim. If this is not unique we
    # do not guess — leave the original conv in place rather than risk rebinding a
    # module whose runtime input is not per-patch shaped (fast_forward assumes one
    # flattened patch per row).
    patch = [c for c in convs if c[1].in_channels <= 4]
    if len(patch) != 1:
        logger.warning(
            "[%s] patch-embed rebind skipped: expected exactly one RGB-input (in_channels<=4) "
            "vision conv, found %d among %s",
            label,
            len(patch),
            [n for n, _ in convs],
        )
        return

    name, conv = patch[0]
    out_ch = conv.out_channels
    spatial_dims = len(conv.kernel_size)
    in_features = conv.in_channels * math.prod(conv.kernel_size)
    weight_2d = conv.weight.reshape(out_ch, in_features)  # view; shares storage, no copy
    bias = conv.bias

    def fast_forward(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        out = F.linear(x.reshape(n, -1).to(weight_2d.dtype), weight_2d, bias)
        return out.reshape(n, out_ch, *([1] * spatial_dims))

    # Validate equivalence on a small synthetic patch batch at load time (off the
    # request path) before swapping. The op is mathematically a matmul over the
    # flattened patch, so any divergence beyond bf16 rounding means the shape
    # assumption is wrong — keep the original conv if so.
    with torch.inference_mode():
        probe = torch.randn(8, conv.in_channels, *conv.kernel_size, device=conv.weight.device, dtype=conv.weight.dtype)
        ref = conv(probe).float()
        got = fast_forward(probe).float().reshape_as(ref)
    max_abs = (ref - got).abs().max().item()
    # The conv and the matmul are the same operation, so they differ only by
    # low-precision (bf16) rounding — small relative to the output scale. A genuine
    # shape mismatch instead produces differences on the order of the outputs
    # themselves. Tolerance scaled to the output magnitude separates the two
    # cleanly without a magic absolute constant.
    tol = 1e-2 + 0.05 * ref.abs().max().item()
    if not math.isfinite(max_abs) or max_abs > tol:
        logger.warning(
            "[%s] patch-embed rebind skipped: %s output mismatch (max_abs_diff=%.3e tol=%.3e)",
            label,
            name,
            max_abs,
            tol,
        )
        return

    conv.forward = fast_forward  # ty: ignore[invalid-assignment]
    logger.info(
        "[%s] rebound vision patch-embed %s (%s, kernel=%s) to F.linear (in=%d out=%d, max_abs_diff=%.3e)",
        label,
        name,
        type(conv).__name__,
        tuple(conv.kernel_size),
        in_features,
        out_ch,
        max_abs,
    )
