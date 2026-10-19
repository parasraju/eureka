import time
from typing import Any

import httpx
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .base import LLMRequest, LLMResponse, ProviderConfig, generate_request_id
from .rate_limiter import rate_limited_complete, rate_limited_embed, get_global_limiter, set_global_limiter


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

        # Use rate-limited complete if global limiter is set
        from .rate_limiter import rate_limited_complete as _rl_complete

        response = await _rl_complete(
            self,
            req,
            rate_limiter=get_global_limiter(),
        )

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
        # Use rate-limited embed if global limiter is set
        from .rate_limiter import rate_limited_embed as _rl_embed

        return await _rl_embed(
            self,
            texts,
            model,
            rate_limiter=get_global_limiter(),
        )

    def cost_usd(self, resp: LLMResponse) -> float:
        model = resp.model.lower()
        if "gpt-4o-mini" in model:
            input_cost = 0.15 / 1_000_000
            output_cost = 0.60 / 1_000_000
        elif "gpt-4o" in model:
            input_cost = 2.50 / 1_000_000
            output_cost = 10.00 / 1_000_000
        elif "gpt-4-turbo" in model or "gpt-4-1106" in model:
            input_cost = 10.00 / 1_000_000
            output_cost = 30.00 / 1_000_000
        elif "gpt-3.5-turbo" in model:
            input_cost = 0.50 / 1_000_000
            output_cost = 1.50 / 1_000_000
        else:
            input_cost = 2.50 / 1_000_000
            output_cost = 10.00 / 1_000_000
        return (resp.input_tokens * input_cost) + (resp.output_tokens * output_cost)

    async def list_models(self) -> list[str]:
        if self._model_cache is not None:
            return self._model_cache
        try:
            models = await self.client.models.list()
            self._model_cache = [m.id for m in models.data]
            return self._model_cache
        except Exception:
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]

    async def health_check(self) -> bool:
        try:
            await self.client.chat.completions.create(
                model="gpt-4o-mini",
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
