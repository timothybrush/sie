"""Tests for SGLang embedding adapter (HTTP server mode)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.sglang import _server
from sie_server.adapters.sglang._server import STARTUP_TIMEOUT_S as _STARTUP_TIMEOUT_S
from sie_server.adapters.sglang._server import parse_device_index
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestSGLangEmbeddingAdapter:
    """Tests for SGLangEmbeddingAdapter with mocked HTTP server."""

    @pytest.fixture
    def adapter(self) -> "SGLangEmbeddingAdapter":
        """Create an adapter instance."""
        return SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            normalize=True,
            max_seq_length=8192,
            mem_fraction_static=0.85,
            query_template="Instruct: {instruction}\nQuery:{text}",
            default_instruction="Given a query, retrieve relevant passages",
        )

    def test_capabilities(self) -> None:
        """Adapter reports correct capabilities."""
        adapter = SGLangEmbeddingAdapter("test-model")
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense"]

    def test_load_contract_flags(self) -> None:
        assert SGLangEmbeddingAdapter.requires_main_thread is False
        assert SGLangEmbeddingAdapter.manages_own_load_timeout is True

    def test_startup_timeout_prefers_explicit_profile_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.setenv(name, "120")

        assert _server.resolve_startup_timeout(1800) == 1800

    def test_startup_timeout_accepts_operator_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SIE_MODEL_READY_TIMEOUT_S", "1234")

        assert _server.resolve_startup_timeout() == 1234

    def test_startup_timeout_rejects_non_finite_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SIE_MODEL_READY_TIMEOUT_S", "inf")
        monkeypatch.setenv("SIE_SERVER_STARTUP_TIMEOUT_S", "1200")

        assert _server.resolve_startup_timeout(float("inf")) == 1200

    def test_startup_timeout_rejects_liveness_budget_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SIE_WORKER_LIVENESS_BUDGET_S", "900")

        with pytest.raises(ValueError, match="SIE_WORKER_LIVENESS_BUDGET_S"):
            _server.resolve_startup_timeout(900)

    def test_dims_before_load_returns_none(self) -> None:
        """Dims returns None before first encode."""
        adapter = SGLangEmbeddingAdapter("test-model")
        assert adapter.dims.dense is None

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load starts SGLang server subprocess."""
        # Setup mocks
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            mem_fraction_static=0.85,
            compute_precision="bfloat16",
        )
        adapter.load("cuda:0")

        # Verify Popen was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert "python" in cmd[0]
        assert "-m" in cmd
        assert "sglang.launch_server" in cmd
        assert "--model-path" in cmd
        assert "Qwen/Qwen3-Embedding-8B" in cmd
        assert "--is-embedding" in cmd
        assert "--port" in cmd
        assert "30000" in cmd
        assert "--dtype" in cmd
        assert "bfloat16" in cmd

        # Verify health check was called
        mock_requests_get.assert_called()

        # Verify server URL is set
        assert adapter._server_url == "http://localhost:30000"

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_different_device(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load parses device index and sets CUDA_VISIBLE_DEVICES."""
        mock_find_port.return_value = 30001
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter("test-model")
        adapter.load("cuda:1")

        # Check that CUDA_VISIBLE_DEVICES was set to 1
        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("CUDA_VISIBLE_DEVICES") == "1"

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode(self, mock_post: MagicMock) -> None:
        """Encode returns dense embeddings via HTTP."""
        # Setup mock response - OpenAI-compatible format
        embeddings = _RNG.standard_normal((2, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": embeddings[0].tolist(), "index": 0},
                {"embedding": embeddings[1].tolist(), "index": 1},
            ]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=False)
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense"])

        assert output.batch_size == 2
        assert output.dense is not None
        assert output.dense[0].shape == (4096,)

        # Verify HTTP call (now uses OpenAI-compatible endpoint)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:30000/v1/embeddings"
        assert call_args[1]["json"]["input"] == ["hello", "world"]

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_normalizes(self, mock_post: MagicMock) -> None:
        """Encode normalizes embeddings when configured."""
        # Setup mock with non-normalized embeddings - OpenAI-compatible format
        embeddings = [3.0, 4.0, 0.0] + [0.0] * 4093
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings, "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=True)
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        output = adapter.encode(items, output_types=["dense"])

        # Check normalization (3-4-5 triangle -> norm of 5)
        norm = np.linalg.norm(output.dense[0])
        assert abs(norm - 1.0) < 1e-5

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_with_query_template(self, mock_post: MagicMock) -> None:
        """Encode applies query template for queries."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            query_template="Instruct: {instruction}\nQuery:{text}",
            default_instruction="search",
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="hello")]
        adapter.encode(items, output_types=["dense"], is_query=True)

        # Check that the formatted text was passed (now uses "input" key)
        call_args = mock_post.call_args
        texts = call_args[1]["json"]["input"]
        assert texts == ["Instruct: search\nQuery:hello"]

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_with_instruction_override(self, mock_post: MagicMock) -> None:
        """Encode uses provided instruction over default."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            query_template="Instruct: {instruction}\nQuery:{text}",
            default_instruction="default",
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="hello")]
        adapter.encode(items, output_types=["dense"], instruction="custom", is_query=True)

        call_args = mock_post.call_args
        texts = call_args[1]["json"]["input"]
        assert texts == ["Instruct: custom\nQuery:hello"]

    @patch("sie_server.adapters.sglang._server.os.getpgid")
    @patch("sie_server.adapters.sglang._server.os.killpg")
    def test_unload(self, mock_killpg: MagicMock, mock_getpgid: MagicMock) -> None:
        """Unload stops the SGLang server subprocess."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        mock_getpgid.return_value = 12345  # Return same as pid

        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._process = mock_process
        adapter._server_url = "http://localhost:30000"
        adapter.unload()

        # Verify process was terminated
        mock_killpg.assert_called()
        mock_getpgid.assert_called_with(12345)
        assert adapter._server_url is None
        assert adapter._process is None

    def test_encode_before_load_raises(self) -> None:
        """Encode before load raises error."""
        adapter = SGLangEmbeddingAdapter("test-model")
        items = [Item(text="test")]

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_unsupported_output_type_raises(self, mock_post: MagicMock) -> None:
        """Encode raises for unsupported output types."""
        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_requires_text(self, mock_post: MagicMock) -> None:
        """Encode raises for items without text."""
        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._server_url = "http://localhost:30000"

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text"):
            adapter.encode(items, output_types=["dense"])

    def test_memory_footprint_before_load(self) -> None:
        """Memory footprint returns 0 before load."""
        adapter = SGLangEmbeddingAdapter("test-model")
        assert adapter.memory_footprint() == 0

    def test_memory_footprint_after_load(self) -> None:
        """Memory footprint returns 0 (uses system monitoring)."""
        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._server_url = "http://localhost:30000"
        adapter._process = MagicMock()

        # SGLang adapter returns 0 because we rely on system GPU memory monitoring
        assert adapter.memory_footprint() == 0

    def test_parse_device_index(self) -> None:
        """Device index parsing works correctly.

        ``_parse_device_index`` moved to the shared ``_server`` module as
        :func:`parse_device_index` when the adapter package was split.
        """
        assert parse_device_index("cuda") == 0
        assert parse_device_index("cuda:0") == 0
        assert parse_device_index("cuda:1") == 1
        assert parse_device_index("cuda:7") == 7
        assert parse_device_index("cpu") == 0

    @patch("sie_server.adapters.sglang._server.os.getpgid")
    @patch("sie_server.adapters.sglang._server.os.killpg")
    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    @patch("sie_server.adapters.sglang._server.time.monotonic")
    def test_load_timeout_raises(
        self,
        mock_monotonic: MagicMock,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
        mock_killpg: MagicMock,
        mock_getpgid: MagicMock,
    ) -> None:
        """Load raises if server fails to start within timeout."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process running but not healthy
        mock_process.stdout = MagicMock()
        mock_process.stdout.read.return_value = b"startup failed"
        mock_popen.return_value = mock_process
        mock_getpgid.return_value = 12345  # For cleanup

        # Simulate timeout
        mock_monotonic.side_effect = [0, _STARTUP_TIMEOUT_S + 1]
        mock_requests_get.side_effect = Exception("Connection refused")

        adapter = SGLangEmbeddingAdapter("test-model")
        with pytest.raises(RuntimeError, match="failed to start"):
            adapter.load("cuda:0")

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_detects_dimension(self, mock_post: MagicMock) -> None:
        """First encode detects embedding dimension."""
        embeddings = _RNG.standard_normal((1, 2048)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=False)
        adapter._server_url = "http://localhost:30000"

        # Dimension not set initially
        assert adapter._dense_dim is None

        items = [Item(text="test")]
        adapter.encode(items, output_types=["dense"])

        # Dimension detected after first encode
        assert adapter._dense_dim == 2048
        assert adapter.dims.dense == 2048

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_rejects_configured_dense_dim_mismatch(self, mock_post: MagicMock) -> None:
        """Configured dense_dim mismatches fail before NumPy row assignment."""
        embeddings = _RNG.standard_normal((1, 2048)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=False, dense_dim=1024)
        adapter._server_url = "http://localhost:30000"

        with pytest.raises(ValueError, match="configured dense_dim=1024, observed=2048"):
            adapter.encode([Item(text="test")], output_types=["dense"])


class TestSGLangLoRA:
    """Tests for SGLang LoRA support."""

    def test_lora_paths_init(self) -> None:
        """Adapter accepts lora_paths at init."""
        lora_paths = {"legal": "org/legal-lora", "medical": "/path/to/medical"}
        adapter = SGLangEmbeddingAdapter("test-model", lora_paths=lora_paths)

        assert adapter._lora_paths == lora_paths
        assert adapter.available_loras == ["legal", "medical"]
        assert adapter.lora_enabled is True

    def test_lora_disabled_by_default(self) -> None:
        """LoRA is disabled when no lora_paths provided."""
        adapter = SGLangEmbeddingAdapter("test-model")

        assert adapter._lora_paths == {}
        assert adapter.available_loras == []
        assert adapter.lora_enabled is False

    def test_max_loras_per_batch_init(self) -> None:
        """Adapter accepts max_loras_per_batch at init."""
        adapter = SGLangEmbeddingAdapter("test-model", max_loras_per_batch=16)
        assert adapter._max_loras_per_batch == 16

    def test_max_loras_per_batch_default(self) -> None:
        """max_loras_per_batch defaults to 8."""
        adapter = SGLangEmbeddingAdapter("test-model")
        assert adapter._max_loras_per_batch == 8

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_with_lora(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load adds LoRA flags when lora_paths is provided."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="test-model",
            lora_paths={"legal": "org/legal-lora", "medical": "/path/to/medical"},
            max_loras_per_batch=4,
        )
        adapter.load("cuda:0")

        # Verify command includes LoRA flags
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]

        assert "--enable-lora" in cmd
        assert "--lora-paths" in cmd
        # Find the lora-paths value
        lora_idx = cmd.index("--lora-paths")
        lora_paths_value = cmd[lora_idx + 1]
        assert "legal=" in lora_paths_value
        assert "medical=" in lora_paths_value

        assert "--max-loras-per-batch" in cmd
        batch_idx = cmd.index("--max-loras-per-batch")
        assert cmd[batch_idx + 1] == "4"

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_without_lora_no_flags(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load does not add LoRA flags when lora_paths is empty."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter(model_name_or_path="test-model")
        adapter.load("cuda:0")

        cmd = mock_popen.call_args[0][0]
        assert "--enable-lora" not in cmd
        assert "--lora-paths" not in cmd
        assert "--max-loras-per-batch" not in cmd

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_with_lora(self, mock_post: MagicMock) -> None:
        """Encode uses LoRA adapter name when set_active_lora() is called first."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            lora_paths={"legal": "org/legal-lora"},
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.set_active_lora("legal")
        adapter.encode(items, output_types=["dense"])

        # Verify LoRA name is used as model name in request
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "legal"

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_without_lora_uses_default(self, mock_post: MagicMock) -> None:
        """Encode uses 'default' model name when no lora is specified."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            lora_paths={"legal": "org/legal-lora"},
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.encode(items, output_types=["dense"])

        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "default"

    def test_encode_invalid_lora_raises(self) -> None:
        """Encode raises ValueError for unknown LoRA adapter set via set_active_lora."""
        adapter = SGLangEmbeddingAdapter(
            "test-model",
            lora_paths={"legal": "org/legal-lora"},
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.set_active_lora("unknown")
        with pytest.raises(ValueError, match="LoRA 'unknown' not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_lora_without_any_loaded_raises(self) -> None:
        """Encode raises ValueError when LoRA set but none loaded."""
        adapter = SGLangEmbeddingAdapter("test-model")  # No lora_paths
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.set_active_lora("legal")
        with pytest.raises(ValueError, match=r"LoRA 'legal' not loaded.*Available: \[\]"):
            adapter.encode(items, output_types=["dense"])
