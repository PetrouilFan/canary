"""Embedding backends.

Local-first: Qwen3-Embedding-0.6B via ONNX (no torch) is the default; an
OpenAI-compatible endpoint and fastembed are alternatives; when nothing is
available retrieval degrades to tag + recency scoring (spec 4.14) instead of
blocking startup on a network fetch.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

from canary.core.config import Config
from canary.core.observability import Log


class EmbeddingUnavailable(RuntimeError):
    pass


class EmbeddingBackend:
    name = "none"
    dim = 0

    @property
    def available(self) -> bool:
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailable(self.name)


class UnavailableBackend(EmbeddingBackend):
    def __init__(self, reason: str = "no embedding backend configured"):
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailable(self.reason)


class HashBackend(EmbeddingBackend):
    """Deterministic local vectors for tests and offline dev. No model, no net."""

    name = "hash"

    def __init__(self, dim: int = 64):
        self.dim = dim

    @property
    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed)
            vec = [rng.uniform(-1, 1) for _ in range(self.dim)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def onnx_repo_for(model: str) -> str:
    if "/" in model and model.split("/")[0] == "onnx-community":
        return model
    base = model.split("/")[-1]
    return f"onnx-community/{base}-ONNX"


def model_slug(model: str) -> str:
    return model.replace("/", "__")


class OnnxBackend(EmbeddingBackend):
    """Qwen-style causal embedding model exported to ONNX.

    Last-token pooling over the attention mask, then L2 normalization
    (the pooling recipe from the Qwen3-Embedding model card).
    """

    name = "onnx"

    def __init__(self, model_dir: Path, onnx_file: str, max_length: int, dim: int | None = None):
        import onnxruntime  # noqa: F401  (import error is handled by resolve_backend)
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / onnx_file
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="<|endoftext|>")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self._input_specs = [(i.name, i.type, i.shape) for i in self.session.get_inputs()]
        self._inputs = {name for name, _type, _shape in self._input_specs}
        shape = self.session.get_outputs()[0].shape
        inferred = shape[-1] if isinstance(shape[-1], int) else None
        self.dim = int(dim or inferred or 1024)

    @property
    def available(self) -> bool:
        return True

    def _empty_kv_inputs(self, batch: int) -> dict[str, Any]:
        """Zero-length past_key_values for exports that require them."""
        import numpy as np

        feed: dict[str, Any] = {}
        for name, type_name, shape in self._input_specs:
            if not name.startswith("past_key_values"):
                continue
            dtype = np.float32 if "float" in type_name else np.int64
            dims = [batch if d == "batch_size" else (d if isinstance(d, int) else 0) for d in shape]
            feed[name] = np.zeros(tuple(dims), dtype=dtype)
        return feed

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import numpy as np

        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        if "position_ids" in self._inputs:
            feed["position_ids"] = np.tile(
                np.arange(input_ids.shape[1], dtype=np.int64), (input_ids.shape[0], 1)
            )
        feed.update(self._empty_kv_inputs(input_ids.shape[0]))
        outputs = self.session.run(None, feed)
        hidden = outputs[0]
        last = attention.sum(axis=1) - 1
        pooled = hidden[np.arange(hidden.shape[0]), np.clip(last, 0, None)]
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (pooled / norms).tolist()


class OpenAIBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, base_url: str, api_key_env: str, model: str, dim: int | None = None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.dim = int(dim or 0)
        self._client = httpx.Client(timeout=60.0)

    @property
    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        key = os.environ.get(self.api_key_env, "")
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        data.sort(key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") or [] for item in data]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors


class FastEmbedBackend(EmbeddingBackend):
    name = "fastembed"

    def __init__(self, model: str):
        from fastembed import TextEmbedding

        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        chosen = model if model in supported else "sentence-transformers/all-MiniLM-L6-v2"
        self.model = chosen
        self._model = TextEmbedding(model_name=chosen)
        self.dim = 384
        for m in TextEmbedding.list_supported_models():
            if m["model"] == chosen:
                self.dim = int(m["dim"])
                break

    @property
    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self._model.embed(texts)]


# ---------------------------------------------------------------------------
# resolution / download
# ---------------------------------------------------------------------------

def _onnx_files(config: Config) -> tuple[str, str]:
    onnx_file = config.get("embedding.onnx_file", "onnx/model_quantized.onnx")
    repo = config.get("embedding.onnx_repo") or onnx_repo_for(config.get("embedding.model"))
    return repo, onnx_file


def onnx_ready(config: Config) -> bool:
    cache = Path(config.get("embedding.cache_dir"))
    slug = model_slug(config.get("embedding.model"))
    _, onnx_file = _onnx_files(config)
    return (cache / slug / onnx_file).exists() and (cache / slug / "tokenizer.json").exists()


def download_onnx_model(config: Config, log: Log | None = None, force: bool = False) -> Path:
    """Fetch ONNX weights + tokenizer into the configured cache dir."""
    from huggingface_hub import snapshot_download

    cache = Path(config.get("embedding.cache_dir"))
    slug = model_slug(config.get("embedding.model"))
    target = cache / slug
    repo, onnx_file = _onnx_files(config)
    if onnx_ready(config) and not force:
        return target
    if log:
        log.info("embedding_download_start", repo=repo, file=onnx_file, target=str(target))
    patterns = [
        onnx_file,
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "*.txt",
        "*.json",
    ]
    try:
        snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=patterns)
    except Exception as exc:
        raise EmbeddingUnavailable(f"download failed: {exc}") from exc
    if log:
        log.info("embedding_download_done", repo=repo, target=str(target))
    return target


def can_import_onnx() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_backend(
    config: Config,
    log: Log | None = None,
    *,
    allow_download: bool | None = None,
) -> EmbeddingBackend:
    mode = str(config.get("embedding.backend", "auto") or "auto").lower()
    model = config.get("embedding.model")

    if mode == "hash":
        return HashBackend(dim=int(config.get("embedding.hash_dim") or 64))
    if mode == "none":
        return UnavailableBackend("embedding backend disabled (embedding.backend=none)")

    if mode in ("auto", "onnx"):
        if can_import_onnx():
            cache = Path(config.get("embedding.cache_dir"))
            slug = model_slug(model)
            target = cache / slug
            if not onnx_ready(config):
                want = (
                    allow_download
                    if allow_download is not None
                    else bool(config.get("embedding.download"))
                )
                if want:
                    try:
                        download_onnx_model(config, log)
                    except EmbeddingUnavailable as exc:
                        if log:
                            log.warn("embedding_download_failed", error=str(exc))
                elif log:
                    log.warn(
                        "embedding_not_cached",
                        model=model,
                        hint="run `canary init` to pre-download",
                    )
            if onnx_ready(config):
                try:
                    return OnnxBackend(
                        target,
                        onnx_file=_onnx_files(config)[1],
                        max_length=int(config.get("embedding.max_length") or 512),
                        dim=config.get("embedding.dim"),
                    )
                except Exception as exc:
                    if log:
                        log.warn("embedding_onnx_failed", error=str(exc))
        elif mode == "onnx" and log:
            log.warn("embedding_onnx_unavailable", detail="onnxruntime/tokenizers not installed")

    openai_cfg = config.get("embedding.openai") or {}
    openai_url = openai_cfg.get("base_url")
    if openai_url and mode in ("auto", "openai"):
        return OpenAIBackend(
            base_url=openai_url,
            api_key_env=openai_cfg.get("api_key_env") or "HARNESS_MODEL_API_KEY",
            model=openai_cfg.get("model") or model,
            dim=config.get("embedding.dim"),
        )

    if mode in ("auto", "fastembed"):
        try:
            return FastEmbedBackend(model)
        except Exception as exc:
            if mode == "fastembed" and log:
                log.warn("embedding_fastembed_failed", error=str(exc))
            if mode == "auto" and log:
                log.info("embedding_degraded", error=str(exc))

    return UnavailableBackend("no embedding backend available; retrieval degrades to tag + recency")


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
