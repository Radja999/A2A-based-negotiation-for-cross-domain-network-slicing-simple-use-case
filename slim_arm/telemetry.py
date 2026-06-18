"""OTel setup for slim_arm — in-memory exporter for local dev."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_span_exporter: InMemorySpanExporter | None = None
_tracer_provider: TracerProvider | None = None


def setup_tracing(service_name: str) -> trace.Tracer:
    """Initialize OTel with in-memory export. Call once per process."""
    global _span_exporter, _tracer_provider

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _span_exporter = exporter
    _tracer_provider = provider

    return trace.get_tracer(service_name)


def get_recent_spans(limit: int = 50) -> list[dict]:
    """Return the most recent finished spans as dicts for the dashboard."""
    if _span_exporter is None:
        return []
    spans = _span_exporter.get_finished_spans()
    result = []
    for span in spans[-limit:]:
        result.append({
            "name":        span.name,
            "start_ms":    span.start_time // 1_000_000,
            "end_ms":      span.end_time   // 1_000_000,
            "duration_ms": (span.end_time - span.start_time) // 1_000_000,
            "status":      span.status.status_code.name,
            "attributes":  dict(span.attributes or {}),
        })
    return result


def get_tracer(name: str = "slim_arm") -> trace.Tracer:
    return trace.get_tracer(name)
