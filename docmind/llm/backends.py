"""The three real backends plus a fake one for tests. Each is ~30 lines on purpose."""

from __future__ import annotations

import time

import httpx

from docmind.config import Settings, get_settings
from docmind.llm.base import LLM, LLMResponse, estimate_cost


class AzureOpenAILLM:
    name = "azure"

    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str) -> None:
        if not api_key or not endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")
        from openai import AzureOpenAI

        self.model = deployment
        self._client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )

    def complete(
        self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
            if usage
            else 0.0,
            latency_s=time.perf_counter() - started,
        )


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key)

    def complete(
        self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0
    ) -> LLMResponse:
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cost_usd=estimate_cost(
                self.model, message.usage.input_tokens, message.usage.output_tokens
            ),
            latency_s=time.perf_counter() - started,
        )


class OllamaLLM:
    """Local model through Ollama's HTTP API (http://localhost:11434). Costs nothing."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: float = 300.0) -> None:
        self.model = model
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def complete(
        self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            cost_usd=0.0,
            latency_s=time.perf_counter() - started,
        )


class FakeLLM:
    """Echoes a canned answer; records the prompts so tests can inspect them."""

    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str = "FAKE ANSWER [doc.pdf, p. 1]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(
        self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0
    ) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(self.reply, self.model, len(user) // 4, len(self.reply) // 4, 0.0, 0.0)


def get_llm(settings: Settings | None = None) -> LLM:
    """Build the backend named by LLM_BACKEND."""
    s = settings or get_settings()
    if s.llm_backend == "azure":
        return AzureOpenAILLM(
            s.azure_openai_api_key,
            s.azure_openai_endpoint,
            s.azure_openai_deployment,
            s.azure_openai_api_version,
        )
    if s.llm_backend == "anthropic":
        return AnthropicLLM(s.anthropic_api_key, s.anthropic_model)
    return OllamaLLM(s.ollama_base_url, s.ollama_model)
