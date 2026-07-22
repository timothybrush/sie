"""Backport SGLang's Transformers 5 LightOnOCR config normalization.

SGLang 0.5.10 reads ``vision_config.rope_theta`` while Transformers 5 stores
the same value in ``vision_config.rope_parameters``. Upstream SGLang fixed this
in #28292. This site hook runs only in the OCR engine subprocess and can be
removed when the shared bundle advances to a release containing that fix.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any

if os.environ.get("SIE_SGLANG_LIGHTON_OCR_COMPAT") == "1":
    from sglang.srt.models.lightonocr import (  # ty: ignore[unresolved-import]
        LightOnOCRForConditionalGeneration,
    )

    if not getattr(LightOnOCRForConditionalGeneration, "_sie_rope_parameters_compat", False):
        _original_init = LightOnOCRForConditionalGeneration.__init__

        @wraps(_original_init)
        def _compat_init(self: Any, *, config: Any, prefix: str = "", **kwargs: Any) -> None:
            vision_config = config.vision_config
            config_dict = vision_config.to_dict()
            rope_parameters = getattr(vision_config, "rope_parameters", None)
            if "rope_theta" not in config_dict and isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
                if rope_theta is not None:
                    vision_config.rope_theta = rope_theta
            _original_init(self, config=config, prefix=prefix, **kwargs)

        LightOnOCRForConditionalGeneration.__init__ = _compat_init
        LightOnOCRForConditionalGeneration._sie_rope_parameters_compat = True
