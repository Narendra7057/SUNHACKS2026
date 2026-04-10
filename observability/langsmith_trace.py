import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def trace_step(step_name: str, input_data: Any, output_data: Any) -> None:
    """Visible trace helper that prints reasoning flow for demo observability."""
    print(f"[LangSmith Trace] {step_name}: {input_data} -> {output_data}")


class LangSmithTracer:
    """Thin LangSmith wrapper with local JSONL fallback audit logging."""

    def __init__(self, project_name: str = "TrustGuard-AI"):
        self.project_name = project_name
        self.client = None
        self.enabled = False

        logs_dir = Path(__file__).resolve().parents[1] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.local_trace_file = logs_dir / "langsmith_trace.jsonl"

        try:
            from langsmith import Client

            if os.getenv("LANGCHAIN_API_KEY"):
                self.client = Client()
                self.enabled = True
        except Exception:
            self.client = None
            self.enabled = False

    def trace_step(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record step data to LangSmith when configured, and always to local JSONL."""
        trace_step(step_name, inputs, outputs)
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "project": self.project_name,
            "step": step_name,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
        }

        with self.local_trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if not self.enabled or self.client is None:
            return

        try:
            self.client.create_run(
                name=f"TrustGuard-{step_name}",
                run_type="chain",
                inputs=inputs,
                outputs=outputs,
                extra=metadata or {},
                project_name=self.project_name,
            )
        except Exception:
            # Keep demo resilient even when external tracing is unavailable.
            pass
