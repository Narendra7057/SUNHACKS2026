from crewai import Agent

from agents.contract_agent import build_ollama_llm


class GovernanceAgent:
    """CrewAI supervisor agent that finalizes governance action from risk signals."""

    def __init__(self):
        self.llm = build_ollama_llm()
        self.agent = Agent(
            role="Governance Supervisor",
            goal="Decide governance action (BLOCK/REVIEW/ALLOW) using risk, anomaly, and policy signals.",
            backstory="You are the compliance supervisor layer over worker AI agents.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )
