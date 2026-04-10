"""Domain-specific agent factory for multi-domain governance."""

from typing import Dict, Any
from crewai import Agent
from .contract_agent import ContractAnalysisAgent


class DomainAgentFactory:
    """Factory to create domain-specific agents with unique personas and prompts."""

    DOMAIN_CONFIGS = {
        "legal": {
            "name": "Legal Compliance Agent",
            "role": "Legal Documents Analyzer",
            "goal": "Analyze contract clauses for legal risks, liability exposure, and fairness",
            "description": "You are an expert legal compliance analyst specializing in contract review, risk assessment, and contract fairness evaluation.",
        },
        "ecommerce": {
            "name": "E-commerce Product Agent",
            "role": "Product Recommendation & Fraud Detection",
            "goal": "Evaluate product recommendations and detect fraudulent or low-quality items",
            "description": "You are an expert e-commerce analyst specializing in product quality assessment, recommendation fairness, and fraud detection.",
        },
        "fintech": {
            "name": "Fintech Loan Agent",
            "role": "Loan Risk Assessment",
            "goal": "Evaluate loan applications and assess credit risk, fraud indicators, and policy compliance",
            "description": "You are an expert financial risk analyst specializing in loan assessment, credit risk evaluation, and regulatory compliance.",
        },
        "healthcare": {
            "name": "Healthcare Diagnostic Agent",
            "role": "Medical Diagnosis & Treatment Recommendation",
            "goal": "Analyze patient symptoms and suggest potential diagnoses while ensuring medical accuracy",
            "description": "You are a medical AI assistant specializing in symptom analysis, diagnosis suggestion, and treatment recommendations.",
        },
    }

    @staticmethod
    def get_domain_prompt(domain: str, input_data: str) -> str:
        """Generate domain-specific analysis prompt."""
        prompts = {
            "legal": f"""Analyze the following contract clause for legal risks:

{input_data}

Return JSON with:
{{
    "decision": "ACCEPTABLE or RISKY",
    "reasoning": "Brief explanation",
    "key_risks": ["..."],
    "liability_exposure": "Low/Medium/High"
}}

Focus on:
- Liability clauses
- Fairness and balance
- One-sided terms
- Business risk exposure""",

            "ecommerce": f"""Analyze the following product for recommendation quality and fraud indicators:

{input_data}

Return JSON with:
{{
    "decision": "ACCEPTABLE or RISKY",
    "reasoning": "Brief explanation",
    "quality_issues": ["..."],
    "fraud_indicators": "None/Low/High"
}}

Focus on:
- Product quality
- Pricing anomalies
- Seller reputation
- Fraud patterns""",

            "fintech": f"""Analyze the following loan application for risk:

{input_data}

Return JSON with:
{{
    "decision": "ACCEPTABLE or RISKY",
    "reasoning": "Brief explanation",
    "credit_risk": "Low/Medium/High",
    "fraud_indicators": ["..."]
}}

Focus on:
- Credit history
- Debt-to-income ratio
- Fraud indicators
- Compliance requirements""",

            "healthcare": f"""Analyze the following patient information for diagnostic suggestions:

{input_data}

Return JSON with:
{{
    "decision": "ACCEPTABLE or RISKY",
    "reasoning": "Brief explanation",
    "potential_diagnoses": ["..."],
    "urgency_level": "Low/Medium/High"
}}

Focus on:
- Symptom analysis
- Risk factors
- Recommended tests
- Medical urgency""",
        }
        return prompts.get(domain, prompts["legal"])

    @staticmethod
    def get_risk_evaluation_prompt(domain: str, input_data: str) -> str:
        """Generate domain-specific risk evaluation prompt."""
        prompts = {
            "legal": f"""Evaluate risk in contract clause:

{input_data}

Return JSON with:
{{
    "risk_level": "LOW, MODERATE, or HIGH",
    "reason": "Detailed reason"
}}

Assess: liability, fairness, compliance impact""",

            "ecommerce": f"""Evaluate product recommendation risk:

{input_data}

Return JSON with:
{{
    "risk_level": "LOW, MODERATE, or HIGH",
    "reason": "Detailed reason"
}}

Assess: quality, fraud, user impact""",

            "fintech": f"""Evaluate loan approval risk:

{input_data}

Return JSON with:
{{
    "risk_level": "LOW, MODERATE, or HIGH",
    "reason": "Detailed reason"
}}

Assess: credit risk, fraud, regulatory compliance""",

            "healthcare": f"""Evaluate diagnostic recommendation risk:

{input_data}

Return JSON with:
{{
    "risk_level": "LOW, MODERATE, or HIGH",
    "reason": "Detailed reason"
}}

Assess: medical accuracy, patient safety, urgency""",
        }
        return prompts.get(domain, prompts["legal"])

    @staticmethod
    def get_domain_config(domain: str) -> Dict[str, str]:
        """Get configuration for a specific domain."""
        return DomainAgentFactory.DOMAIN_CONFIGS.get(domain, DomainAgentFactory.DOMAIN_CONFIGS["legal"])

    @staticmethod
    def get_domain_markers(domain: str) -> list:
        """Get domain-specific markers for rule-based detection."""
        markers = {
            "legal": [
                "not liable",
                "no liability",
                "disclaims all liability",
                "without notice",
                "terminate immediately",
                "entire risk",
                "at your own risk",
            ],
            "ecommerce": [
                "counterfeit",
                "damaged",
                "defective",
                "fake review",
                "suspicious seller",
                "unauthorized reseller",
            ],
            "fintech": [
                "high debt",
                "bankruptcy",
                "default",
                "missing documentation",
                "identity mismatch",
                "fraud flag",
            ],
            "healthcare": [
                "critical condition",
                "emergency",
                "severe symptoms",
                "life-threatening",
                "allergic reaction",
                "overdose",
            ],
        }
        return markers.get(domain, markers["legal"])

    @staticmethod
    def is_valid_domain(domain: str) -> bool:
        """Check if domain is valid."""
        return domain.lower() in DomainAgentFactory.DOMAIN_CONFIGS
