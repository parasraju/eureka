import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass
class LLMRequest:
    model: str
    messages: list[dict]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    seed: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    request_id: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: dict
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class LLMProvider(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...

    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...

    def cost_usd(self, resp: LLMResponse) -> float: ...

    async def list_models(self) -> list[str]: ...

    async def health_check(self) -> bool: ...


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    extra_headers: dict = field(default_factory=dict)


def generate_request_id() -> str:
    return str(uuid.uuid4())
