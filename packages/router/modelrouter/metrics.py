"""Optional Prometheus exporter: call record(resp, tenant) after each route()."""
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "router_requests_total",
    "Total router requests",
    ["tenant", "tier", "model"],
)
COST = Counter(
    "router_cost_usd_total",
    "Total USD spent",
    ["tenant", "model"],
)
LATENCY = Histogram(
    "router_latency_ms",
    "End-to-end latency in ms",
    ["tier", "model"],
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000),
)


def record(resp, tenant: str) -> None:
    REQUESTS.labels(tenant=tenant, tier=resp.tier, model=resp.model).inc()
    COST.labels(tenant=tenant, model=resp.model).inc(resp.cost_usd)
    LATENCY.labels(tier=resp.tier, model=resp.model).observe(resp.latency_ms)