import os
import urllib.request
from typing import Any

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from pydantic import Field, PrivateAttr

try:
    from langchain.llms import Ollama
except Exception:  # LangChain>=0.1 community split fallback
    from langchain_community.llms import Ollama


class OllamaCrewAIAdapter(BaseLLM):
    """CrewAI BaseLLM adapter that delegates real inference to LangChain Ollama."""

    llm_type: str = "ollama"
    model: str = "mistral"
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str = "http://localhost:11434"
    provider: str = "ollama"
    prefer_upload: bool = False
    is_litellm: bool = False
    stop: list[str] | None = None
    additional_params: dict[str, Any] = Field(default_factory=dict)

    _backend: Any = PrivateAttr()

    def _healthcheck(self) -> None:
        """Fail fast when Ollama runtime is unavailable."""
        tags_url = f"{self.base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=3) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama healthcheck failed with status {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Ollama is not running at {self.base_url}. Start Ollama before running TrustGuard AI."
            ) from exc

    def model_post_init(self, __context: Any) -> None:
        self._healthcheck()
        self._backend = Ollama(model=self.model, base_url=self.base_url)

    def _stringify_messages(self, messages: str | list[Any]) -> str:
        if isinstance(messages, str):
            return messages

        content_parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                content_parts.append(str(message.get("content", "")))
            else:
                content_parts.append(str(getattr(message, "content", message)))
        return "\n".join(part for part in content_parts if part)

    def call(
        self,
        messages: str | list[Any],
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        prompt = self._stringify_messages(messages)
        response = self._backend.invoke(prompt)
        return response if isinstance(response, str) else str(response)


def build_ollama_llm() -> OllamaCrewAIAdapter:
    """Shared real Ollama LLM instance (mistral) for all agents."""
    model_name = os.getenv("OLLAMA_MODEL", "mistral")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaCrewAIAdapter(model=model_name, base_url=base_url)


class ContractAnalysisAgent:
    """CrewAI contract analysis agent using Ollama backend."""

    def __init__(self):
        self.llm = build_ollama_llm()
        self.agent = Agent(
            role="Contract Analysis Specialist",
            goal="Classify if a clause is ACCEPTABLE or RISKY with concise rationale.",
            backstory="You perform first-pass legal risk triage for contract governance teams.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )
