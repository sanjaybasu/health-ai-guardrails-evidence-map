"""Three-vendor structured-output call layer with a response cache, a cost ledger, and budget stops.

Every call returns parsed JSON that satisfies the supplied JSON schema. Responses are cached on disk by a hash of
(vendor, model, system, user, schema), so re-running a step never pays twice for the same prompt.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
CACHE = ROOT / "data" / "cache" / "llm"
LEDGER = ROOT / "state" / "cost_ledger.jsonl"
SECRETS = Path("~/.config/ebm-verify/secrets.env").expanduser()

_clients: dict = {}
_run_spend = 0.0


class BudgetExceeded(RuntimeError):
    pass


def _key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def client(vendor: str):
    if vendor not in _clients:
        if vendor == "claude":
            import anthropic
            _clients[vendor] = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"), max_retries=4, timeout=900.0)
        elif vendor == "openai":
            import openai
            _clients[vendor] = openai.OpenAI(api_key=_key("OPENAI_API_KEY"), max_retries=4, timeout=900.0)
        elif vendor == "gemini":
            from google import genai
            _clients[vendor] = genai.Client(api_key=_key("GEMINI_API_KEY"))
        else:
            raise ValueError(vendor)
    return _clients[vendor]


# ----------------------------------------------------------------------------- budget

def _price(model: str, tin: int, tout: int) -> float:
    p = CFG["prices_per_mtok"].get(model, {"in": 10.0, "out": 40.0})
    return (tin * p["in"] + tout * p["out"]) / 1e6


def month_spend() -> float:
    if not LEDGER.exists():
        return 0.0
    month = dt.date.today().strftime("%Y-%m")
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        r = json.loads(line)
        if r["ts"].startswith(month):
            total += r["usd"]
    return total


def _check_budget():
    b = {"per_run_usd": float(os.environ.get("GEM_RUN_CAP", CFG["budget"]["per_run_usd"])),
         "monthly_usd": float(os.environ.get("GEM_MONTH_CAP", CFG["budget"]["monthly_usd"]))}
    if _run_spend >= b["per_run_usd"]:
        raise BudgetExceeded(f"per-run cap ${b['per_run_usd']} reached (${_run_spend:.2f})")
    if month_spend() >= b["monthly_usd"]:
        raise BudgetExceeded(f"monthly cap ${b['monthly_usd']} reached")


def _record(step, vendor, model, tin, tout, secs):
    global _run_spend
    usd = _price(model, tin, tout)
    _run_spend += usd
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "step": step, "vendor": vendor,
                            "model": model, "in": tin, "out": tout, "usd": round(usd, 5), "secs": secs}) + "\n")


def run_spend() -> float:
    return _run_spend


# ----------------------------------------------------------------------------- vendor calls

def _claude(model, system, user, schema, effort, max_tokens):
    c = client("claude")
    kw = dict(model=model, max_tokens=max_tokens,
              system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
              messages=[{"role": "user", "content": user}],
              output_config={"format": {"type": "json_schema", "schema": schema}, "effort": effort})
    with c.messages.stream(**kw) as s:
        msg = s.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("claude refusal")
    text = next(b.text for b in msg.content if b.type == "text")
    u = msg.usage   # cache reads bill at ~10% and writes at 125% of input; fold into an input-equivalent count
    tin = u.input_tokens + int(0.1 * (u.cache_read_input_tokens or 0)) + int(1.25 * (u.cache_creation_input_tokens or 0))
    return json.loads(text), tin, u.output_tokens


def _openai(model, system, user, schema, effort, max_tokens):
    c = client("openai")
    r = c.responses.create(model=model, instructions=system, input=user, max_output_tokens=max_tokens,
                           reasoning={"effort": effort},
                           text={"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}})
    return json.loads(r.output_text), r.usage.input_tokens, r.usage.output_tokens


def _gemini(model, system, user, schema, effort, max_tokens):
    from google.genai import types
    c = client("gemini")
    level = {"low": "low", "medium": "medium", "high": "high"}.get(effort, "high")
    cfg = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                      response_json_schema=schema, max_output_tokens=max_tokens,
                                      thinking_config=types.ThinkingConfig(thinking_level=level))
    r = c.models.generate_content(model=model, contents=user, config=cfg)
    if not r.text:
        raise json.JSONDecodeError("empty gemini response", "", 0)
    u = r.usage_metadata
    tout = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return json.loads(r.text), u.prompt_token_count or 0, tout


_CALL = {"claude": _claude, "openai": _openai, "gemini": _gemini}


def call(vendor: str, model: str, system: str, user: str, schema: dict, *, step: str,
         effort: str = "high", max_tokens: int = 32000, use_cache: bool = True) -> dict:
    """Return schema-valid JSON from one vendor. Cached by content hash; retried on transient failure."""
    h = hashlib.sha256(json.dumps([vendor, model, system, user, schema, effort], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{h}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())["out"]
    _check_budget()
    last = None
    for attempt in range(4):
        t0 = time.time()
        try:
            out, tin, tout = _CALL[vendor](model, system, user, schema, effort, max_tokens)
            _record(step, vendor, model, tin, tout, round(time.time() - t0, 1))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"vendor": vendor, "model": model, "step": step, "out": out}))
            tmp.replace(path)
            return out
        except (json.JSONDecodeError, StopIteration) as e:   # malformed output; retry once more
            last = e
        except Exception as e:  # network, 429, 5xx: back off
            last = e
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"{vendor}/{model} failed after retries: {last!r}")


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
