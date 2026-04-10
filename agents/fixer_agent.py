from crewai import Agent

from agents.contract_agent import build_ollama_llm


class FixerAgent:
    """CrewAI correction agent using Ollama backend."""

    def __init__(self):
        self.llm = build_ollama_llm()
        self.agent = Agent(
            role="Decision Correction Specialist",
            goal=(
                "Apply strict governance correction with non-regression: never reduce risk severity; "
                "keep or escalate only."
            ),
            backstory=(
                "You are a strict AI governance system. Always prioritize compliance and safety. "
                "If liability, unfair risk, or anomaly exists, classify as HIGH RISK."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )
