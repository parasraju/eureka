import time

import httpx
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .base import LLMRequest, LLMResponse, ProviderConfig, generate_request_id


class GoogleProvider:
    name = "google"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = genai.Client(
            api_key=config.api_key,
            http_options=types.HttpOptions(
                base_url=config.base_url,
                timeout=config.timeout * 1000,
            ),
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
        system_instruction = None
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0]["content"]
            messages = messages[1:]

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        config = types.GenerateContentConfig(
            max_output_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            system_instruction=system_instruction,
        )

        response = await self.client.aio.models.generate_content(
            model=req.model,
            contents=contents,
            config=config,
        )

        latency = time.perf_counter() - start

        text = response.text or ""

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        raw = response.model_dump()

        return LLMResponse(
            text=text,
            model=response.model_version or req.model,
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
        response = await self.client.aio.models.embed_content(
            model=model,
            contents=texts,
        )
        return [emb.values for emb in response.embeddings]

    def cost_usd(self, resp: LLMResponse) -> float:
        model = resp.model.lower()
        if "gemini-1.5-pro" in model:
            input_cost = 1.25 / 1_000_000
            output_cost = 5.00 / 1_000_000
        elif "gemini-1.5-flash" in model:
            input_cost = 0.075 / 1_000_000
            output_cost = 0.30 / 1_000_000
        elif "gemini-1.0-pro" in model:
            input_cost = 0.50 / 1_000_000
            output_cost = 1.50 / 1_000_000
        else:
            input_cost = 0.50 / 1_000_000
            output_cost = 1.50 / 1_000_000
        return (resp.input_tokens * input_cost) + (resp.output_tokens * output_cost)

    async def list_models(self) -> list[str]:
        if self._model_cache is not None:
            return self._model_cache
        try:
            models = []
            async for model in self.client.aio.models.list():
                name = getattr(model, "name", "") or ""
                if name:
                    models.append(name.split("/")[-1])
            self._model_cache = models
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
            await self.client.aio.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text="ping")])],
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return True
        except Exception:
            return False


def create_google_provider(api_key: str, **kwargs) -> GoogleProvider:
    config = ProviderConfig(
        name="google",
        api_key=api_key,
        base_url=kwargs.get("base_url", "https://generativelanguage.googleapis.com"),
        timeout=kwargs.get("timeout", 60.0),
        max_retries=kwargs.get("max_retries", 3),
    )
    return GoogleProvider(config)
