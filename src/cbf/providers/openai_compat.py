import time
from typing import Any

import httpx
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .base import LLMRequest, LLMResponse, ProviderConfig, generate_request_id


class OpenAICompatibleProvider:
    name = "openai_compat"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )
        self._model_cache: list[str] | None = None
        self._model_error: str | None = None

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
    )
    async def complete(self, req: LLMRequest) -> LLMResponse:
        request_id = generate_request_id()
        start = time.perf_counter()

        messages = req.messages
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.seed is not None:
            kwargs["seed"] = req.seed
        kwargs.update(req.extra)

        response = await self.client.chat.completions.create(**kwargs)

        latency = time.perf_counter() - start
        choice = response.choices[0]
        text = choice.message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

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
        response = await self.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    def _model_rates(self, model: str) -> tuple[float, float, str | None]:
        model = model.lower()
        if "gpt-4o-mini" in model:
            return 0.15 / 1_000_000, 0.60 / 1_000_000, None
        if "gpt-4o" in model:
            return 2.50 / 1_000_000, 10.00 / 1_000_000, None
        if "gpt-4-turbo" in model or "gpt-4-1106" in model:
            return 10.00 / 1_000_000, 30.00 / 1_000_000, None
        if "gpt-3.5-turbo" in model:
            return 0.50 / 1_000_000, 1.50 / 1_000_000, None
        return 2.50 / 1_000_000, 10.00 / 1_000_000, "assumed-gpt-4o"

    def cost_usd(self, resp: LLMResponse) -> float:
        input_cost, output_cost, _ = self._model_rates(resp.model)
        return (resp.input_tokens * input_cost) + (resp.output_tokens * output_cost)

    def cost_basis(self, resp: LLMResponse) -> str | None:
        return self._model_rates(resp.model)[2]

    async def list_models(self) -> list[str]:
        if self._model_cache is not None:
            return self._model_cache
        try:
            models = await self.client.models.list()
            self._model_cache = [m.id for m in models.data]
            return self._model_cache
        except Exception as e:
            self._model_error = str(e)
            return []

    async def health_check(self, model: str | None = None) -> bool:
        if model is None:
            models = await self.list_models()
            if not models:
                detail = f": {self._model_error}" if self._model_error else ""
                raise RuntimeError(f"No models available for provider {self.name}{detail}")
            model = models[0]
        try:
            await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None, **kwargs):
        config = ProviderConfig(
            name="openai",
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            **kwargs,
        )
        super().__init__(config)


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def __init__(self, api_key: str, base_url: str | None = None, **kwargs):
        config = ProviderConfig(
            name="openrouter",
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://github.com/creative-benchmark-finder", "X-Title": "CBF"},
            **kwargs,
        )
        super().__init__(config)

    def cost_usd(self, resp: LLMResponse) -> float:
        return super().cost_usd(resp) * 1.1
