"""Local-only client shims for running Graphiti against on-node resources.

Graphiti needs three model clients: an LLM (entity/edge extraction + dedupe), an
embedder, and a cross-encoder (reranker). On this HPC node we have:

  * a local OpenAI-compatible vLLM chat endpoint (Qwen2.5-7B) -- but it does NOT
    honour the OpenAI ``json_schema`` response_format, only ``json_object``;
  * a cached ``BAAI/bge-base-en-v1.5`` sentence-transformer (768-d) -- but no
    embeddings endpoint and no cached reranker model.

So we wrap the stock ``OpenAIGenericClient`` in ``json_object`` mode and supply a
sentence-transformers embedder plus a cosine reranker built from the same model.
Nothing here reaches the public internet.
"""

from __future__ import annotations

import asyncio
import json
import typing
from collections.abc import Iterable

import numpy as np
import openai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from graphiti_core.llm_client.openai_generic_client import DEFAULT_MODEL, OpenAIGenericClient
from graphiti_core.prompts.models import Message

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768


class LocalSTEmbedder(EmbedderClient):
    """sentence-transformers embedder (bge-base, L2-normalised for cosine)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = EMBED_DIM

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=float).tolist()

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        texts = [input_data] if isinstance(input_data, str) else list(input_data)
        vecs = await asyncio.to_thread(self._encode, [str(t) for t in texts])
        return vecs[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []
        return await asyncio.to_thread(self._encode, list(input_data_list))


class LocalCosineReranker(CrossEncoderClient):
    """Rerank passages by cosine similarity to the query using the bge embedder.

    Avoids downloading a dedicated cross-encoder (``bge-reranker-v2-m3`` is not
    cached) and avoids spending extra LLM calls on reranking.
    """

    def __init__(self, embedder: LocalSTEmbedder):
        self.embedder = embedder

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []
        vecs = await self.embedder.create_batch([query, *passages])
        q = np.asarray(vecs[0])
        scores = [float(np.dot(q, np.asarray(p))) for p in vecs[1:]]
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return ranked


class CountingLLMClient(OpenAIGenericClient):
    """OpenAIGenericClient that counts how many chat completions Graphiti issues.

    We force ``structured_output_mode='json_object'`` because this vLLM server
    ignores ``json_schema`` response_format (returns prose).
    """

    def __init__(self, config: LLMConfig, max_tokens: int = 2048):
        super().__init__(
            config=config,
            max_tokens=max_tokens,
            structured_output_mode="json_object",
        )
        # Qwen2.5-7B is served with max_model_len=8192. Graphiti issues some calls
        # with its 16384-token default, which overflows the context window; clamp
        # every completion so prompt + completion stays under the limit.
        self.completion_cap = max_tokens
        self.n_calls = 0

    async def generate_response(self, *args, **kwargs):  # type: ignore[override]
        self.n_calls += 1
        return await super().generate_response(*args, **kwargs)

    async def _generate_response(  # type: ignore[override]
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        """Like the parent, but uses vLLM's native ``guided_json`` (constrained
        decoding) instead of the OpenAI ``json_object`` response_format.

        This server ignores ``json_schema`` response_format and, in plain
        ``json_object`` mode, pads valid JSON with whitespace up to ``max_tokens``
        (tens of seconds per call). ``guided_json`` constrains output to the
        schema and stops at the closing brace -- correct and ~1-2 orders of
        magnitude faster.
        """
        max_tokens = min(max_tokens, self.completion_cap)
        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ("user", "system"):
                openai_messages.append({"role": m.role, "content": m.content})

        # Use exactly one guided-decoding mode: guided_json when we have a schema,
        # else plain json_object. vLLM rejects specifying both.
        kwargs: dict[str, typing.Any] = {}
        if response_model is not None:
            kwargs["extra_body"] = {"guided_json": response_model.model_json_schema()}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            result = response.choices[0].message.content or ""
            if not result:
                raise EmptyResponseError("LLM returned an empty response")
            return json.loads(self._strip_code_fences(result))
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception:
            raise


def make_llm_client(base_url: str, api_key: str, model: str, max_tokens: int = 2048):
    cfg = LLMConfig(api_key=api_key, model=model, base_url=base_url, max_tokens=max_tokens)
    return CountingLLMClient(cfg, max_tokens=max_tokens)


__all__ = [
    "LocalSTEmbedder",
    "LocalCosineReranker",
    "CountingLLMClient",
    "make_llm_client",
    "Message",
    "EMBED_DIM",
]
