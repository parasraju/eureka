import os
import time
import httpx
from typing import Any
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from .base import LLMProvider, LLMRequest, LLMResponse, ProviderConfig, generate_request_id


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )
        self._model_cache: list[str] | None = None

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
    )
    async def complete(self, req: LLMRequest) -> LLMResponse:
        request_id = generate_request_id()
        start = time.perf_counter()

        messages = req.messages
        system_prompt = None
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.seed is not None:
            kwargs["seed"] = req.seed
        kwargs.update(req.extra)

        response = await self.client.messages.create(**kwargs)

        latency = time.perf_counter() - start

        text = ""
        if response.content:
            for block in response.content:
                if block.type == "text":
                    text += block.text

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        raw = response.model_dump()

        return LLMResponse(
            text=text,
            model=response.model,
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            raw=raw,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
    )
    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError("Anthropic does not provide embeddings API. Use OpenAI-compatible or sentence-transformers.")

    def cost_usd(self, resp: LLMResponse) -> float:
        model = resp.model.lower()
        if "claude-3-5-sonnet" in model:
            input_cost = 3.00 / 1_000_000
            output_cost = 15.00 / 1_000_000
        elif "claude-3-5-haiku" in model:
            input_cost = 0.80 / 1_000_000
            output_cost = 4.00 / 1_000_000
        elif "claude-3-opus" in model:
            input_cost = 15.00 / 1_000_000
            output_cost = 75.00 / 1_000_000
        elif "claude-3-sonnet" in model:
            input_cost = 3.00 / 1_000_000
            output_cost = 15.00 / 1_000_000
        elif "claude-3-haiku" in model:
            input_cost = 0.25 / 1_000_000
            output_cost = 1.25 / 1_000_000
        else:
            input_cost = 3.00 / 1_000_000
            output_cost = 15.00 / 1_000_000
        return (resp.input_tokens * input_cost) + (resp.output_tokens * output_cost)

    async def list_models(self) -> list[str]:
        if self._model_cache is not None:
            return self._model_cache
        self._model_cache = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
        return self._model_cache

    async def health_check(self) -> bool:
        try:
            await self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False


def create_anthropic_provider(api_key: str, **kwargs) -> AnthropicProvider:
    config = ProviderConfig(
        name="anthropic",
        api_key=api_key,
        base_url=kwargs.get("base_url", "https://api.anthropic.com"),
        timeout=kwargs.get("timeout", 60.0),
        max_retries=kwargs.get("max_retries", 3),
    )
    return AnthropicProvider(config)