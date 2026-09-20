"""python -m modelrouter report [spend.jsonl]   -> summary of a spend log
   python -m modelrouter route "text" [--tenant t] [--litellm]   -> route one request"""
import argparse
import json
import sys

from modelrouter.router import ModelRouter, RouteRequest, spend_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="modelrouter")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("log", nargs="?", default="spend.jsonl")
    q = sub.add_parser("route")
    q.add_argument("text")
    q.add_argument("--tenant", default="default")
    q.add_argument("--log", default="spend.jsonl")
    q.add_argument("--litellm", action="store_true", help="call the real model through LiteLLM (needs provider API keys)")
    a = p.parse_args(argv)
    if a.cmd == "report":
        print(json.dumps(spend_report(a.log), indent=2))
        return 0
    caller = None
    if a.litellm:
        from modelrouter.callers import litellm_caller

        caller = litellm_caller()
    resp = ModelRouter(caller=caller, spend_log_path=a.log).route(RouteRequest("cli", a.text, tenant=a.tenant))
    print(json.dumps({"model": resp.model, "tier": resp.tier, "cost_usd": resp.cost_usd, "cost_source": resp.cost_source, "text": resp.text[:200]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
