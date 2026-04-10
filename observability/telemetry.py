import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


class TelemetryLogger:
    """OpenTelemetry logger for governance pipeline events."""

    def __init__(self, service_name: str = "trustguard-governance"):
        self.service_name = service_name

        logs_dir = Path(__file__).resolve().parents[1] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / "telemetry.log"

        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        current_provider = trace.get_tracer_provider()
        if not isinstance(current_provider, TracerProvider):
            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)

        self.tracer = trace.get_tracer(service_name)

    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Write both OpenTelemetry span attributes and file logs for observability."""
        with self.tracer.start_as_current_span(event_name) as span:
            for key, value in payload.items():
                span.set_attribute(key, str(value))

        line = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_name,
            "payload": payload,
        }
        self.logger.info(json.dumps(line, ensure_ascii=False, default=str))
