from typing import Dict, Optional


class RuleValidator:
    """Rule-based governance validator."""

    def evaluate(self, clause: str, agent_decision: str) -> Dict[str, object]:
        """Check deterministic risk rules for contract clauses and return structured issue data."""
        # Normalize text once so phrase detection is consistent.
        clause_lower = clause.lower()
        issue: Optional[Dict[str, str]] = None

        # Liability phrases that should always force HIGH RISK treatment.
        liability_phrases = [
            "not liable",
            "no liability",
            "not responsible",
            "no responsibility",
        ]

        matched_phrase = next((phrase for phrase in liability_phrases if phrase in clause_lower), None)
        if matched_phrase:
            keyword = matched_phrase
            start = clause_lower.find(keyword)
            end = start + len(keyword)
            issue = {
                "type": "rule_violation",
                "what_is_wrong": "Liability/responsibility disclaimer indicates elevated legal risk but was not treated as HIGH RISK.",
                "where_is_wrong": f"Text segment '{clause[start:end]}' at character range {start}-{end}.",
                "why_it_is_wrong": "Clauses denying liability can shift all loss responsibility and must be classified conservatively.",
            }

        # Any detected liability phrase must be classified as HIGH RISK.
        wrong_decision = issue is not None and agent_decision.upper() != "HIGH RISK"

        return {
            "error": wrong_decision,
            "issue": issue,
            "risk_level": "HIGH" if issue else "LOW",
        }
