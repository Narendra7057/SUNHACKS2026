from crewai import Agent

from agents.contract_agent import build_ollama_llm


class RiskEvaluationAgent:
    """CrewAI semantic risk evaluation agent using Ollama backend."""

    def __init__(self):
        self.llm = build_ollama_llm()
        self.agent = Agent(
            role="Risk Evaluation Specialist",
            goal="Evaluate semantic risk level (LOW/MODERATE/HIGH) with reasoning.",
            backstory="You detect fairness issues, liability removal, and one-sided clauses.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )
