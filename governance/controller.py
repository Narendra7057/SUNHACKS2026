import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from crewai import Crew, Task

from agents.contract_agent import ContractAnalysisAgent
from agents.explainer_agent import ExplainerAgent
from agents.fixer_agent import FixerAgent
from agents.governance_agent import GovernanceAgent
from agents.validator_agent import RiskEvaluationAgent
from agents.domain_agent_factory import DomainAgentFactory
from governance.anomaly_detector import OutputAnomalyDetector
from observability.langsmith_trace import LangSmithTracer, trace_step
from observability.telemetry import TelemetryLogger


def _parse_json_output(raw_output: str, step_name: str) -> Dict[str, Any]:
    """Parse JSON returned by an agent task; raise if invalid to avoid fake outputs."""
    text = (raw_output or "").strip()
    decoder = json.JSONDecoder()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                # Real LLMs may return unescaped control chars in strings.
                cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate)
                cleaned = cleaned.replace("\r", " ").replace("\n", " ")
                try:
                    return json.loads(cleaned)
                except Exception:
                    pass

        # Fallback: find first decodable JSON object and ignore trailing text.
        for index, char in enumerate(text):
            if char != "{":
                continue
            chunk = text[index:]
            cleaned_chunk = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", chunk)
            cleaned_chunk = cleaned_chunk.replace("\r", " ").replace("\n", " ")
            try:
                parsed, _ = decoder.raw_decode(cleaned_chunk)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        raise ValueError(f"{step_name} produced non-JSON output: {text}") from exc


def analyze_impact(risk_level: str) -> Dict[str, str]:
    if risk_level == "HIGH":
        return {
            "severity": "HIGH",
            "business_risk": "Legal liability exposure",
            "estimated_loss": "High",
        }
    if risk_level == "MODERATE":
        return {
            "severity": "MEDIUM",
            "business_risk": "Potential dispute risk",
            "estimated_loss": "Medium",
        }
    return {
        "severity": "LOW",
        "business_risk": "Minimal risk",
        "estimated_loss": "Low",
    }


def check_compliance(risk_level: str) -> Dict[str, str]:
    if risk_level == "HIGH":
        return {"status": "VIOLATION", "reason": "Clause creates unfair legal risk"}
    if risk_level == "MODERATE":
        return {"status": "WARNING", "reason": "Clause may be legally restrictive"}
    return {"status": "COMPLIANT", "reason": "Clause is acceptable"}


def decide_action(risk_level: str) -> str:
    if risk_level == "HIGH":
        return "BLOCK"
    if risk_level == "MODERATE":
        return "REVIEW"
    return "ALLOW"


RISK_RANK = {
    "LOW": 1,
    "MODERATE": 2,
    "RISKY": 2,
    "HIGH": 3,
    "HIGH RISK": 3,
}


def _rank_risk(label: str) -> int:
    return RISK_RANK.get(str(label or "").strip().upper(), 0)


class GovernanceController:
    """Real governance engine using CrewAI, Ollama, Isolation Forest, LangSmith, and OpenTelemetry."""

    PIPELINE_ORDER = [
        "Agent",
        "Risk",
        "Detection",
        "Governance",
        "Explanation",
        "Audit",
    ]

    def __init__(self):
        self.contract_agent = ContractAnalysisAgent()
        self.risk_agent = RiskEvaluationAgent()
        self.fixer_agent = FixerAgent()
        self.governance_agent = GovernanceAgent()
        self.explainer_agent = ExplainerAgent()

        self.anomaly_detector = OutputAnomalyDetector()
        self.tracer = LangSmithTracer(project_name="TrustGuard-AI")
        self.telemetry = TelemetryLogger(service_name="trustguard-governance")

    def _audit_entry(self, stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "payload": payload,
        }

    def _extract_task_output(self, crew_output: Any, task_name: str, fallback_index: int) -> Dict[str, Any]:
        """Parse a specific task output from CrewOutput.tasks_output using name or index."""
        tasks_output = list(getattr(crew_output, "tasks_output", []) or [])
        selected = None

        for output in tasks_output:
            if str(getattr(output, "name", "")).strip() == task_name:
                selected = output
                break

        if selected is None and tasks_output and fallback_index < len(tasks_output):
            selected = tasks_output[fallback_index]

        if selected is None:
            raise ValueError(f"Crew output missing task result for '{task_name}'")

        raw_text = getattr(selected, "raw", None) or ""
        return _parse_json_output(raw_text, task_name)

    def run_pipeline(self, input_data: str, domain: str = "legal") -> Dict[str, Any]:
        # End-to-end real execution path: CrewAI + Ollama + IsolationForest + LangSmith + OpenTelemetry.
        # Now domain-aware for multi-domain support
        audit_trail: List[Dict[str, Any]] = []
        audit_log: Dict[str, List[Dict[str, Any]]] = {"issues": [], "actions": []}

        # Normalize domain
        domain = domain.lower() if domain else "legal"
        if not DomainAgentFactory.is_valid_domain(domain):
            domain = "legal"

        # Get domain-specific markers for rule-based detection
        domain_markers = DomainAgentFactory.get_domain_markers(domain)
        domain_config = DomainAgentFactory.get_domain_config(domain)

        # 1) Main Crew orchestration: domain analysis + semantic risk evaluation in one kickoff.
        # Use domain-specific prompts
        analysis_prompt = DomainAgentFactory.get_domain_prompt(domain, input_data)
        risk_prompt = DomainAgentFactory.get_risk_evaluation_prompt(domain, input_data)

        contract_task = Task(
            name="analysis",
            description=analysis_prompt,
            expected_output="JSON with decision and reasoning.",
            agent=self.contract_agent.agent,
        )
        risk_task = Task(
            name="risk_evaluation",
            description=risk_prompt,
            expected_output="JSON with risk_level and reason.",
            agent=self.risk_agent.agent,
            context=[contract_task],
        )
        main_crew = Crew(
            agents=[self.contract_agent.agent, self.risk_agent.agent],
            tasks=[contract_task, risk_task],
            verbose=False,
        )
        with self.telemetry.tracer.start_as_current_span("step_analysis_and_risk") as span:
            span.set_attribute("input_length", len(input_data))
            span.set_attribute("domain", domain)
            main_result = main_crew.kickoff()

        original = self._extract_task_output(main_result, "analysis", 0)
        risk_result = self._extract_task_output(main_result, "risk_evaluation", 1)

        trace_step("analysis", {"domain": domain, "input": input_data}, original)
        self.tracer.trace_step("analysis", {"domain": domain, "input": input_data}, original)
        trace_step("risk_evaluation", {"domain": domain, "input": input_data}, risk_result)
        self.tracer.trace_step("risk_evaluation", {"domain": domain, "input": input_data}, risk_result)

        original_decision = str(original.get("decision", "")).strip().upper()
        if original_decision not in {"ACCEPTABLE", "RISKY"}:
            raise ValueError(f"contract_analysis returned invalid decision: {original_decision}")

        original_output = {
            "decision": original_decision,
            "reasoning": str(original.get("reasoning", "No reasoning provided by agent.")).strip(),
            "domain": domain,
        }
        audit_trail.append(self._audit_entry("Agent", original_output))
        self.telemetry.log_event("agent_decision", original_output)
        self.telemetry.logger.info(f"Agent decision ({domain}): {original_decision}")

        risk_level = str(risk_result.get("risk_level", "")).strip().upper()
        if risk_level not in {"LOW", "MODERATE", "HIGH"}:
            raise ValueError(f"risk_evaluation returned invalid risk_level: {risk_level}")
        risk_reason = str(risk_result.get("reason", "Semantic evaluation completed.")).strip()

        # 2) Real anomaly detection (Isolation Forest) from actual extracted text features.
        with self.telemetry.tracer.start_as_current_span("step_anomaly_detection"):
            anomaly_result = self.anomaly_detector.detect(input_data)
        anomaly_detected = bool(anomaly_result["is_anomaly"])
        anomaly_prediction = int(anomaly_result.get("prediction", 1))

        # 3) STRICT governance signal fusion.
        # If anomaly is detected, force HIGH risk.
        if anomaly_detected:
            risk_level = "HIGH"
        system_risk = "HIGH" if (risk_level == "HIGH" or anomaly_prediction == -1) else risk_level
        if anomaly_detected:
            system_risk = "HIGH"
        if anomaly_detected and system_risk == "HIGH":
            risk_reason = f"{risk_reason} Isolation Forest flagged the input as anomalous."

        self.telemetry.log_event("anomaly_detection", anomaly_result)
        self.telemetry.logger.info(f"Anomaly detection result: {anomaly_detected}")

        # 4) Supervisor-layer detection using semantic risk + rules + anomaly.
        liability_markers = domain_markers
        input_lower = input_data.lower()
        rule_triggered = any(marker in input_lower for marker in liability_markers)
        mismatch_detected = original_decision == "ACCEPTABLE" and (system_risk in {"HIGH", "MODERATE"} or rule_triggered)

        error_detected = bool(mismatch_detected or anomaly_detected or (rule_triggered and system_risk != "LOW"))
        issue: Optional[Dict[str, str]] = None
        if error_detected:
            issue = {
                "type": "reasoning_failure",
                "what_is_wrong": "Original agent output is risky or insufficiently strict for compliance governance",
                "where_is_wrong": "Decision classification",
                "why_it_is_wrong": (
                    f"semantic_risk={system_risk}, anomaly_detected={anomaly_detected}, rule_triggered={rule_triggered}. "
                    f"{risk_reason}"
                ),
            }
            audit_log["issues"].append(issue)

        # 4b) CrewAI Supervisor Agent decides final governance action from combined signals.
        governance_task = Task(
            name="governance_supervisor",
            description=(
                "You are the supervisor layer. Return JSON only with keys 'governance_action' and 'supervisor_reason'. "
                "governance_action must be one of BLOCK, REVIEW, ALLOW.\n\n"
                f"Original output: {json.dumps(original_output)}\n"
                f"Risk signals: {json.dumps({'system_risk': system_risk, 'risk_reason': risk_reason, 'anomaly_detected': anomaly_detected, 'rule_triggered': rule_triggered, 'error_detected': error_detected})}\n"
                "Policy mapping preference: HIGH->BLOCK, MODERATE->REVIEW, LOW->ALLOW."
            ),
            expected_output="JSON with governance_action and supervisor_reason.",
            agent=self.governance_agent.agent,
        )
        governance_crew = Crew(
            agents=[self.governance_agent.agent],
            tasks=[governance_task],
            verbose=False,   
        )
        with self.telemetry.tracer.start_as_current_span("step_governance_supervisor"):
            governance_result = governance_crew.kickoff()
        governance_json = self._extract_task_output(governance_result, "governance_supervisor", 0)
        trace_step("governance_supervisor", {"system_risk": system_risk, "anomaly": anomaly_result}, governance_json)
        self.tracer.trace_step("governance_supervisor", {"system_risk": system_risk, "anomaly": anomaly_result}, governance_json)

        llm_action = str(governance_json.get("governance_action", "")).strip().upper()
        governance_action = llm_action if llm_action in {"BLOCK", "REVIEW", "ALLOW"} else decide_action(system_risk)

        # STRICT decision engine: HIGH always BLOCK. Anomaly always BLOCK.
        if anomaly_detected:
            system_risk = "HIGH"
            governance_action = "BLOCK"
        else:
            governance_action = decide_action(system_risk)

        decision_factors: List[str] = []
        if system_risk == "HIGH":
            decision_factors.append("HIGH risk detected")
        if check_compliance(system_risk).get("status") == "VIOLATION":
            decision_factors.append("Compliance violation")
        if anomaly_detected:
            decision_factors.append("Anomaly detected")
        supervisor_reason = str(governance_json.get("supervisor_reason", "Supervisor action applied.")).strip()

        impact = analyze_impact(system_risk)
        compliance = check_compliance(system_risk)

        detection_payload = {
            "risk_level": system_risk,
            "risk_reason": risk_reason,
            "anomaly": anomaly_result,
            "error_detected": error_detected,
            "governance_action": governance_action,
            "supervisor_reason": supervisor_reason,
        }
        audit_trail.append(self._audit_entry("Detection", detection_payload))
        self.telemetry.log_event("risk_evaluation", detection_payload)
        self.telemetry.logger.info(f"Agent Output: {original_output}")
        self.telemetry.logger.info(f"Risk level: {system_risk}")
        self.telemetry.logger.info(f"Governance action: {governance_action}")

        # 5) SELF-REPROMPT: always produce improved governed output.
        # Domain-specific reprompting
        domain_context = {
            "legal": "legal compliance and contract fairness",
            "ecommerce": "product quality and fraud prevention",
            "fintech": "credit risk and fraud detection",
            "healthcare": "medical accuracy and patient safety",
        }
        context_text = domain_context.get(domain, "compliance and risk governance")

        if error_detected and issue:
            reprompt = f"""
You are a strict {domain} compliance AI.

Your previous decision was incorrect.

Re-evaluate the {context_text} considering:

* key risk factors for {domain}
* compliance requirements
* business impact
* compliance

Return JSON only with keys:
{{
    "risk_level": "LOW|MODERATE|HIGH",
    "key_issues": ["..."],
    "final_decision": "HIGH RISK|RISKY|ACCEPTABLE",
    "justification": "..."
}}

Rules:
* NEVER reduce risk level
* You may keep or increase severity only
* Always prioritize compliance and safety
* If liability, unfair risk, or anomaly exists -> classify as HIGH RISK

Correction mode: STRICT. Enforce safer classification when risk signals are high.

Original Output:
{json.dumps(original_output)}

Detected Issue:
{json.dumps(issue)}
""".strip()
        else:
            reprompt = f"""
You are a strict legal compliance AI.

Your previous decision may be acceptable, but generate a governance-refined version.

Re-evaluate the contract considering:

* liability clauses
* fairness
* business risk
* compliance

Return JSON only with keys:
{{
    "risk_level": "LOW|MODERATE|HIGH",
    "key_issues": ["..."],
    "final_decision": "HIGH RISK|RISKY|ACCEPTABLE",
    "justification": "..."
}}

Rules:
* NEVER reduce risk level
* You may keep or increase severity only
* Always prioritize compliance and safety
* If liability, unfair risk, or anomaly exists -> classify as HIGH RISK

Refinement mode: keep correctness, improve clarity, and standardize for audit.

Original Output:
{json.dumps(original_output)}

Input Data:
{input_data}"""

        correction_task = Task(
            name="correction",
            description=corrected_prompt,
            expected_output="JSON with risk_level, key_issues, final_decision, justification, decision, reasoning.",
            agent=self.fixer_agent.agent,
        )
        correction_crew = Crew(
            agents=[self.fixer_agent.agent],
            tasks=[correction_task],
            verbose=False,
        )
        with self.telemetry.tracer.start_as_current_span("step_correction"):
            correction_result = correction_crew.kickoff()
        corrected = self._extract_task_output(correction_result, "correction", 0)

        trace_step("correction", {"issue": issue, "error_detected": error_detected}, corrected)
        self.tracer.trace_step("correction", {"issue": issue, "error_detected": error_detected}, corrected)

        corrected_decision = str(corrected.get("final_decision", original_output["decision"])).strip().upper()
        corrected_justification = str(corrected.get("justification", "Governance refinement applied.")).strip()
        corrected_key_issues = corrected.get("key_issues", [])
        if not isinstance(corrected_key_issues, list):
            corrected_key_issues = [str(corrected_key_issues)]

        corrected_output = {
            "risk_level": str(corrected.get("risk_level", system_risk)).strip().upper(),
            "key_issues": [str(item).strip() for item in corrected_key_issues if str(item).strip()],
            "final_decision": corrected_decision,
            "justification": corrected_justification,
            "decision": corrected_decision,
            "reasoning": corrected_justification,
        }

        # STRICT correction validation: corrected output must not reduce severity.
        original_rank = max(_rank_risk(system_risk), _rank_risk(original_output.get("decision", "")))
        corrected_rank = max(_rank_risk(corrected_output.get("risk_level", "")), _rank_risk(corrected_output.get("decision", "")))
        if corrected_rank < original_rank:
            corrected_output = {
                "risk_level": system_risk,
                "key_issues": ["Correction non-regression safeguard applied"],
                "final_decision": original_output["decision"],
                "justification": original_output["reasoning"],
                "decision": original_output["decision"],
                "reasoning": original_output["reasoning"],
            }

        # Ensure supervised output remains distinct and audit-ready even when classification is unchanged.
        if corrected_output["decision"] == original_output["decision"]:
            corrected_output["reasoning"] = (
                f"Governed refinement: {corrected_output['justification']} | Semantic risk: {system_risk}."
            )
            corrected_output["justification"] = corrected_output["reasoning"]

        comparison = {
            "before": original_output,
            "after": corrected_output,
            "changed": (
                original_output.get("decision") != corrected_output.get("decision")
                or original_output.get("reasoning") != corrected_output.get("reasoning")
            ),
        }

        audit_log["actions"].append(
            {
                "step": "Governance",
                "action": "agent_corrected" if error_detected else "agent_refined",
            }
        )

        audit_trail.append(self._audit_entry("Governance", {"corrected_output": corrected_output}))

        # 6) Explanation generation via CrewAI + Ollama.
        explanation_task = Task(
            name="explanation",
            description=(
                "Generate a clear governance explanation in JSON with key 'explanation'. "
                "Explain why the original decision was right or wrong and what changed.\n\n"
                f"Clause: {clause}\n"
                f"Original decision: {original_output['decision']}\n"
                f"Risk level: {system_risk}\n"
                f"Anomaly detected: {anomaly_detected}\n"
                f"Governance action: {governance_action}\n"
                f"Corrected decision: {corrected_output['decision']}\n"
                f"Issue: {json.dumps(issue) if issue else 'None'}"
            ),
            expected_output="JSON with explanation.",
            agent=self.explainer_agent.agent,
        )
        explanation_crew = Crew(
            agents=[self.explainer_agent.agent],
            tasks=[explanation_task],
            verbose=False,
        )
        with self.telemetry.tracer.start_as_current_span("step_explanation"):
            explanation_result = explanation_crew.kickoff()
        explanation_json = self._extract_task_output(explanation_result, "explanation", 0)
        trace_step("explanation", {"clause": clause}, explanation_json)
        self.tracer.trace_step("explanation", {"clause": clause}, explanation_json)

        explanation = str(explanation_json.get("explanation", "No explanation returned.")).strip()
        if explanation.startswith("{") and explanation.endswith("}"):
            explanation = (
                "The contract was first reviewed by ContractAnalysisAgent. "
                f"Initial result was {original_output['decision']}. "
                f"Governance layer detected risk level {system_risk} with anomaly={anomaly_detected}. "
                f"FixerAgent re-evaluated using a strict compliance reprompt and produced {corrected_output['decision']}."
            )
        if anomaly_detected or check_compliance(system_risk).get("status") == "VIOLATION":
            explanation = (
                f"{explanation} Due to anomaly detection and compliance violation, "
                "the system enforced a stricter governance decision."
            )

        final_payload = {
            "final_decision": corrected_output,
            "governance_action": governance_action,
            "agent_corrected": "FixerAgent",
        }
        self.telemetry.log_event("final_decision", final_payload)
        trace_step("final_decision", {"original_output": original_output, "comparison": comparison}, final_payload)
        self.tracer.trace_step("final_decision", {"original_output": original_output, "comparison": comparison}, final_payload)

        supervisor_audit = {
            "agent": "ContractAgent",
            "corrected_agent": "FixerAgent",
            "original_output": original_output,
            "error_detected": error_detected,
            "reprompt_used": corrected_prompt,
            "corrected_output": corrected_output,
            "comparison": comparison,
            "improvement": "classification corrected" if error_detected else "reasoning standardized for audit",
        }
        audit_log["supervisor"] = supervisor_audit

        audit_trail.append(self._audit_entry("Explanation", {"explanation": explanation}))
        audit_summary = {
            "pipeline_order": self.PIPELINE_ORDER,
            "steps_executed": len(audit_trail),
            "final_status": "corrected" if error_detected else "accepted",
        }
        audit_trail.append(self._audit_entry("Audit", audit_summary))
        audit_log["actions"].append({"step": "Audit", "action": "finalized_audit_log"})

        system_trace = {
            "CrewAI": "Active",
            "Ollama": "mistral",
            "LangSmith": "Tracing enabled",
            "OpenTelemetry": "Logging active",
            "IsolationForest": "Anomaly detection applied",
        }

        return {
            "final_decision": corrected_output,
            "original_output": original_output,
            "risk_level": system_risk,
            "anomaly_detected": anomaly_detected,
            "governance_action": governance_action,
            "decision_factors": decision_factors,
            "corrected_output": corrected_output,
            "explanation": explanation,
            "impact": impact,
            "compliance": compliance,
            "audit_log": audit_log,
            "comparison": comparison,
            "reprompt_used": corrected_prompt,
            "corrected_agent": "FixerAgent",
            "error_detected": error_detected,
            "issues_detected": [issue] if issue else [],
            "trust_score": 0.6 if error_detected else 0.9,
            "system_trace": system_trace,
        }
