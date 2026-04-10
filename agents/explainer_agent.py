from crewai import Agent

from agents.contract_agent import build_ollama_llm


class ExplainerAgent:
    """CrewAI explanation agent using Ollama backend."""

    def __init__(self):
        self.llm = build_ollama_llm()
        self.agent = Agent(
            role="AI Governance Explainer",
            goal="Explain why a decision was wrong and what changed after correction.",
            backstory="You provide concise, auditor-ready governance explanations.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )
