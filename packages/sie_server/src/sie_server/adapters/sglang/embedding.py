"""SGLang adapter for large LLM embedding models (4B+).

SGLang provides memory-efficient inference for LLM embedding models by
pre-allocating KV cache. This prevents OOM under concurrent load that
PyTorch-based adapters can experience with 4B+ models.

Target models:
- Qwen3-Embedding-4B, Qwen3-Embedding-8B
- GTE-Qwen2-7B-instruct
- E5-Mistral-7B-instruct, SFR-Embedding-Mistral
- LLaMA-Embed-Nemotron-8B, NV-Embed-v2

Implementation: Uses SGLang's HTTP server mode (subprocess) rather than the
Engine API to avoid event loop conflicts with uvicorn/uvloop.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters.sglang import _server
from sie_server.core.inference_output import EncodeOutput

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


class SGLangEmbeddingAdapter(BaseAdapter):
    """Adapter for LLM embedding models using SGLang HTTP server backend.

    SGLang pre-allocates GPU memory for the KV cache, providing stable memory
    usage under concurrent load. This is critical for 4B+ LLM embeddings that
    would otherwise OOM with dynamic memory allocation.

    Key differences from PyTorchEmbeddingAdapter:
    - Memory is pre-allocated at load time (controlled by mem_fraction_static)
    - Uses SGLang's HTTP server (subprocess) for inference
    - Supports last-token pooling only (standard for LLM embeddings)

    Note: This adapter starts SGLang as a subprocess server during load().
    Signal handlers in SGLang require main thread execution.

    Example:
        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            mem_fraction_static=0.5,
        )
        adapter.load("cuda:0")
        results = adapter.encode([Item(text="hello")], ["dense"])
    """

    spec = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=("_process", "_server_url", "_dense_dim"))

    # SGLang uses signal handlers that require main thread execution
    requires_main_thread: bool = True

    def _check_loaded(self) -> None:
        if self._server_url is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        mem_fraction_static: float = 0.85,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = True,
        query_template: str | None = None,
        doc_template: str | None = None,
        default_instruction: str | None = None,
        pooling_method: str | None = None,
        lora_paths: dict[str, str] | None = None,
        max_loras_per_batch: int = 8,
        dense_dim: int | None = None,
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        r"""Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings.
            max_seq_length: Maximum sequence length.
            mem_fraction_static: Fraction of GPU memory to pre-allocate (0.0-1.0).
                Lower values leave more headroom for other models. Default 0.85.
            compute_precision: Compute precision (bfloat16 recommended).
            trust_remote_code: Whether to trust remote code in model files.
            query_template: Template for formatting queries. Use {instruction} and
                {text} placeholders. Example: "Instruct: {instruction}\nQuery:{text}"
            doc_template: Template for formatting documents. Use {text} placeholder.
            default_instruction: Default instruction when query_template uses
                {instruction} but none is provided.
            pooling_method: Pooling method for embeddings. Options: "cls", "lasttoken",
                "max", "mean", "mean_sqrt_len_tokens", "weightedmean". If None, uses
                SGLang's default (usually lasttoken for LLM models).
            lora_paths: LoRA adapters to load. Dict mapping adapter name to path.
                Example: {"legal": "org/legal-lora", "medical": "/path/to/medical"}.
                At request time, select via lora parameter in encode().
            max_loras_per_batch: Maximum LoRA adapters per batch. Default 8.
            dense_dim: Configured dense embedding dimension.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._mem_fraction_static = mem_fraction_static
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._query_template = query_template
        self._doc_template = doc_template
        self._default_instruction = default_instruction
        self._pooling_method = pooling_method
        self._lora_paths = lora_paths or {}
        self._max_loras_per_batch = max_loras_per_batch

        self._process: subprocess.Popen[bytes] | None = None
        self._server_url: str | None = None
        self._device: str | None = None
        self._configured_dense_dim: int | None = dense_dim
        self._dense_dim: int | None = dense_dim
        self._active_lora: str | None = None  # Set by set_active_lora() before encode()
        self._output_file: tempfile._TemporaryFileWrapper | None = None

    @property
    def available_loras(self) -> list[str]:
        """Return list of available LoRA adapter names.

        These are the names that can be passed to encode(lora=...).
        """
        return list(self._lora_paths.keys())

    @property
    def lora_enabled(self) -> bool:
        """Return whether LoRA adapters are configured."""
        return bool(self._lora_paths)

    def load(self, device: str) -> None:
        """Load the model by starting SGLang HTTP server as subprocess.

        Args:
            device: Device string (e.g., "cuda:0", "cuda:1").
                    Note: SGLang primarily supports CUDA devices.

        Raises:
            RuntimeError: If server fails to start within timeout.
        """
        self._device = device

        device_index = _server.parse_device_index(device)
        port = _server.find_free_port()
        self._server_url = f"http://localhost:{port}"

        logger.info(
            "Starting SGLang server for %s on device=%s (gpu_id=%d) at port %d",
            self._model_name_or_path,
            device,
            device_index,
            port,
        )

        # Build server command. Use sys.executable so we get the same Python
        # interpreter that has sglang installed (important for uv envs).
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_name_or_path,
            "--is-embedding",
            "--port",
            str(port),
            "--dtype",
            self._compute_precision,
            "--context-length",
            str(self._max_seq_length),
            "--mem-fraction-static",
            str(self._mem_fraction_static),
            "--tp",
            "1",  # Tensor parallel = 1 (single GPU)
            "--log-level",
            "warning",
        ]

        if self._trust_remote_code:
            cmd.append("--trust-remote-code")

        if self._pooling_method:
            cmd.extend(["--pooling-method", self._pooling_method])

        # LoRA configuration for the SGLang server.
        if self._lora_paths:
            cmd.append("--enable-lora")
            lora_path_str = ",".join(f"{name}={path}" for name, path in self._lora_paths.items())
            cmd.extend(["--lora-paths", lora_path_str])
            cmd.extend(["--max-loras-per-batch", str(self._max_loras_per_batch)])
            logger.info(
                "LoRA enabled with %d adapters: %s",
                len(self._lora_paths),
                list(self._lora_paths.keys()),
            )

        self._output_file = _server.open_output_log()
        self._process = _server.launch_sglang_server(cmd, device_index=device_index, output_file=self._output_file)

        if not _server.wait_for_server(self._server_url, self._process, output_file=self._output_file):
            _server.terminate_process(self._process)
            self._process = None
            raise RuntimeError(_server.ERR_SERVER_STARTUP)

        logger.info(
            "SGLang server ready: %s at %s",
            self._model_name_or_path,
            self._server_url,
        )

    def unload(self) -> None:
        """Unload the model by stopping SGLang server subprocess."""
        if self._process is not None:
            logger.info("Shutting down SGLang server for %s", self._model_name_or_path)
            _server.terminate_process(self._process)
            self._process = None

        self._server_url = None
        self._device = None
        self._dense_dim = self._configured_dense_dim

    def memory_footprint(self) -> int:
        """Return the GPU memory usage in bytes.

        SGLang pre-allocates memory in the subprocess. Return 0 and let the
        registry use actual GPU memory monitoring instead.
        """
        return 0

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        Args:
            items: List of items to encode.
            output_types: Which outputs to return (only "dense" supported).
            instruction: Optional instruction for queries.
            is_query: Whether items are queries (True) or documents (False).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.

        Raises:
            ValueError: If active LoRA is not loaded.

        Note:
            LoRA is set via set_active_lora() called by the worker before encode().
        """
        self._check_loaded()

        # Validate active LoRA if specified
        lora = self._active_lora
        if lora is not None and lora not in self._lora_paths:
            available = list(self._lora_paths.keys()) if self._lora_paths else []
            msg = f"LoRA '{lora}' not loaded. Available: {available}"
            raise ValueError(msg)

        self._validate_output_types(output_types)

        # Resolve runtime options (config defaults -> profile -> request overrides)
        # Note: pooling is NOT overridable for SGLang (set at subprocess startup via --pooling-method)
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)
        default_instruction = opts.get("default_instruction", self._default_instruction)
        normalize = opts.get("normalize", self._normalize)

        texts = self._format_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            default_instruction=default_instruction,
        )

        # SGLang rejects empty/whitespace-only inputs, so we need to:
        # 1. Track which indices have empty text
        # 2. Only send non-empty texts to SGLang
        # 3. Insert zero vectors for empty items in the result
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # If all texts are empty, return zero vectors
        if not non_empty_texts:
            if self._dense_dim is None:
                msg = "SGLang embedding dimension is unknown; configure tasks.encode.dense.dim"
                raise RuntimeError(msg)
            dim = self._dense_dim
            embeddings = np.zeros((len(items), dim), dtype=np.float32)
            return EncodeOutput(
                dense=embeddings,
                batch_size=len(items),
                is_query=is_query,
                dense_dim=dim,
            )

        # Call SGLang HTTP API (OpenAI-compatible embeddings endpoint)
        # When LoRA is specified, use it as the model name; otherwise use "default"
        model_name = lora if lora is not None else "default"
        response = requests.post(
            f"{self._server_url}/v1/embeddings",
            json={
                "model": model_name,
                "input": non_empty_texts,
                "encoding_format": "float",
            },
            timeout=60,
        )
        if response.status_code != 200:
            logger.error(
                "SGLang error %d for %d texts: %s",
                response.status_code,
                len(non_empty_texts),
                response.text[:500],
            )
        response.raise_for_status()
        result = response.json()

        # Extract embeddings from OpenAI-format result
        # Response format: {"data": [{"embedding": [...], "index": 0}, ...]}
        non_empty_embeddings = self._extract_embeddings(result, len(non_empty_texts))

        dense_dim = self._validate_or_set_dense_dim(non_empty_embeddings)

        # Normalize if configured
        if normalize:
            non_empty_embeddings = self._normalize_embeddings(non_empty_embeddings)

        # Reconstruct full result array with zero vectors for empty inputs
        embeddings = np.zeros((len(items), dense_dim), dtype=np.float32)
        for result_idx, original_idx in enumerate(non_empty_indices):
            embeddings[original_idx] = non_empty_embeddings[result_idx]

        return EncodeOutput(
            dense=embeddings,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=dense_dim,
        )

    def _validate_or_set_dense_dim(self, embeddings: np.ndarray) -> int:
        """Validate observed SGLang embedding width against configured dense_dim."""
        if embeddings.ndim != 2:
            msg = f"SGLang server returned embeddings with invalid shape {embeddings.shape}"
            raise ValueError(msg)
        observed_dim = embeddings.shape[1]
        if self._configured_dense_dim is not None and observed_dim != self._configured_dense_dim:
            msg = (
                "SGLang embedding dimension mismatch: "
                f"configured dense_dim={self._configured_dense_dim}, observed={observed_dim}"
            )
            raise ValueError(msg)
        if self._dense_dim is None:
            self._dense_dim = observed_dim
            logger.info("Detected embedding dimension: %d", self._dense_dim)
        elif observed_dim != self._dense_dim:
            msg = f"SGLang embedding dimension changed: expected {self._dense_dim}, observed {observed_dim}"
            raise ValueError(msg)
        assert self._dense_dim is not None
        return self._dense_dim

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. SGLang adapter only supports 'dense'."
            raise ValueError(msg)

    def _format_texts(
        self,
        items: list[Item],
        instruction: str | None,
        *,
        is_query: bool,
        query_template: str | None = None,
        doc_template: str | None = None,
        default_instruction: str | None = None,
    ) -> list[str]:
        r"""Format texts using configured templates.

        For queries with query_template, formats using the template.
        For documents with doc_template, formats using the template.
        Otherwise returns text as-is.
        """
        query_template = query_template if query_template is not None else self._query_template
        doc_template = doc_template if doc_template is not None else self._doc_template
        default_instruction = default_instruction if default_instruction is not None else self._default_instruction
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="SGLangEmbeddingAdapter"))

            text = item.text

            if is_query and query_template:
                # Use provided instruction or default
                instr = instruction or default_instruction or ""
                text = query_template.format(instruction=instr, text=text)
            elif not is_query and doc_template:
                text = doc_template.format(text=text)
            elif instruction:
                # Fallback: prepend instruction if provided but no template
                text = f"{instruction} {text}"

            texts.append(text)
        return texts

    def _extract_embeddings(self, result: dict[str, Any], num_items: int) -> np.ndarray:
        """Extract embeddings from SGLang OpenAI-compatible HTTP response.

        SGLang returns OpenAI-format response:
        {"data": [{"embedding": [...], "index": 0}, ...], "model": "...", "usage": {...}}
        """
        data = result.get("data")
        if not data:
            msg = "SGLang server returned empty response"
            raise RuntimeError(msg)

        if len(data) != num_items:
            msg = f"Expected {num_items} embeddings, got {len(data)}"
            raise RuntimeError(msg)

        # Sort by index to ensure correct order
        data_sorted = sorted(data, key=lambda x: x.get("index", 0))

        # Extract embeddings from each result object
        embeddings_list = []
        for i, item in enumerate(data_sorted):
            embedding = item.get("embedding")
            if embedding is None:
                msg = f"SGLang response item {i} missing 'embedding' key"
                raise RuntimeError(msg)
            embeddings_list.append(embedding)

        # Convert to numpy array [batch, dim]
        embeddings_np = np.array(embeddings_list, dtype=np.float32)

        return embeddings_np

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    # -------------------------------------------------------------------------
    # LoRA Support
    # -------------------------------------------------------------------------

    def supports_lora(self) -> bool:
        """Return True if LoRA adapters are configured."""
        return bool(self._lora_paths)

    def supports_hot_lora_reload(self) -> bool:
        """Return False - SGLang blocks during LoRA loading.

        SGLang's /load_lora_adapter endpoint blocks the server until loading
        completes. This is not true hot-reload like PEFT provides.
        """
        return False

    def set_active_lora(self, lora_name: str | None) -> None:
        """Set the active LoRA for the next encode() call.

        For SGLang, we store the active LoRA and use it as the model name
        in the HTTP request to the SGLang server.

        Args:
            lora_name: LoRA adapter name, or None for base model.
        """
        self._active_lora = lora_name
