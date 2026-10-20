from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from .anthropic import create_anthropic_provider
from .base import LLMProvider, ProviderConfig
from .google import create_google_provider
from .openai_compat import OpenAIProvider, OpenRouterProvider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    google_api_key: str | None = None
    google_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None


settings = Settings()


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._configs: dict[str, ProviderConfig] = {}

    def register(self, name: str, provider: LLMProvider, config: ProviderConfig):
        self._providers[name] = provider
        self._configs[name] = config

    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)

    def get_config(self, name: str) -> ProviderConfig | None:
        return self._configs.get(name)

    def list_names(self) -> list[str]:
        return list(self._providers.keys())

    async def health_check_all(self, models: dict[str, str] | None = None) -> dict[str, bool]:
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check((models or {}).get(name))
            except Exception:
                results[name] = False
        return results


def build_registry_from_settings(cfg: dict[str, Any] | None = None) -> ProviderRegistry:
    registry = ProviderRegistry()
    providers_cfg = cfg.get("providers", {}) if cfg else {}

    if settings.openai_api_key:
        base_url = settings.openai_base_url or providers_cfg.get("openai", {}).get("base_url")
        timeout = providers_cfg.get("openai", {}).get("timeout", 60)
        max_retries = providers_cfg.get("openai", {}).get("max_retries", 3)
        provider = OpenAIProvider(settings.openai_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        config = ProviderConfig(name="openai", api_key=settings.openai_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        registry.register("openai", provider, config)

    if settings.anthropic_api_key:
        base_url = settings.anthropic_base_url or providers_cfg.get("anthropic", {}).get("base_url")
        timeout = providers_cfg.get("anthropic", {}).get("timeout", 60)
        max_retries = providers_cfg.get("anthropic", {}).get("max_retries", 3)
        provider = create_anthropic_provider(settings.anthropic_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        config = ProviderConfig(name="anthropic", api_key=settings.anthropic_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        registry.register("anthropic", provider, config)

    if settings.google_api_key:
        base_url = settings.google_base_url or providers_cfg.get("google", {}).get("base_url")
        timeout = providers_cfg.get("google", {}).get("timeout", 60)
        max_retries = providers_cfg.get("google", {}).get("max_retries", 3)
        provider = create_google_provider(settings.google_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        config = ProviderConfig(name="google", api_key=settings.google_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        registry.register("google", provider, config)

    if settings.openrouter_api_key:
        base_url = settings.openrouter_base_url or providers_cfg.get("openrouter", {}).get("base_url")
        timeout = providers_cfg.get("openrouter", {}).get("timeout", 60)
        max_retries = providers_cfg.get("openrouter", {}).get("max_retries", 3)
        provider = OpenRouterProvider(settings.openrouter_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        config = ProviderConfig(name="openrouter", api_key=settings.openrouter_api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        registry.register("openrouter", provider, config)

    return registry


def parse_model_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Model spec must be 'provider:model', got '{spec}'")
    provider, model = spec.split(":", 1)
    return provider, model
