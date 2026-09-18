"""Load + chaos gate for Track B.

Usage (server must already be running):
    python tests/test_load.py [--base-url http://127.0.0.1:8000]

- 20x sequential POST (same case): report p50/p95, gate p95 < 30s
  (target <=5s cached, <15s cold — warns, doesn't fail, above that).
- 30x mixed valid (cycle public cases): 0x 5xx, valid JSON, 24h plan.
- 1x malformed JSON -> 400 with {"detail":...}, no traceback/secret.
- Local no-key fallback: interpret_notes with empty keys -> no_op, never raises.
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = sys.argv[sys.argv.index("--base-url") + 1] if "--base-url" in sys.argv else "http://127.0.0.1:8000"

pack = json.load(open("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"))
cases = pack["cases"]
failures = 0


def post(payload, timeout=60):
    req = urllib.request.Request(
        f"{BASE_URL}/optimize-energy",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
            dt = time.monotonic() - t0
            return r.status, body, dt, None
    except urllib.error.HTTPError as e:
        dt = time.monotonic() - t0
        try:
            body = json.loads(e.read().decode() or "{}")
        except Exception:
            body = {}
        return e.code, body, dt, None
    except Exception as e:
        dt = time.monotonic() - t0
        return -1, {}, dt, repr(e)


def pct(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = min(len(s) - 1, int(p / 100.0 * len(s)))
    return s[k]


# --- 20x sequential, same case ---
same = cases[0]["input"]
lats = []
for i in range(20):
    st, body, dt, err = post(same)
    lats.append(dt)
    if st != 200:
        print(f"load20 #{i}: FAIL status {st} err={err}")
        failures += 1
p50, p95 = pct(lats, 50), pct(lats, 95)
print(f"20x same-case: p50={p50:.2f}s p95={p95:.2f}s max={max(lats):.2f}s")
if p95 >= 30.0:
    print("FAIL p95 >= 30s (judge timeout)")
    failures += 1
elif p95 > 15.0:
    print("WARN p95 > 15s (1/3 latency pts) — cold LLM likely; cache warming helps")
elif p95 > 5.0:
    print("WARN p95 > 5s (2/3 latency pts)")
else:
    print("OK p95 <= 5s (3/3 latency pts)")

# --- 30x mixed valid ---
bad = 0
for i in range(30):
    payload = cases[i % len(cases)]["input"]
    st, body, dt, err = post(payload)
    if st == -1 or st >= 500:
        print(f"mix30 #{i}: FAIL status={st} err={err}")
        bad += 1
        continue
    if st != 200 or not isinstance(body.get("hourly_plan"), list) or len(body["hourly_plan"]) != 24:
        print(f"mix30 #{i}: FAIL bad shape status={st}")
        bad += 1
if bad:
    print(f"mix30: FAIL {bad}/30 bad")
    failures += 1
else:
    print("mix30: PASS 30/30 valid, 0x 5xx")

# --- malformed JSON -> 400, no trace ---
req = urllib.request.Request(
    f"{BASE_URL}/optimize-energy",
    data=b"{bad json",
    headers={"Content-Type": "application/json"},
)
try:
    urllib.request.urlopen(req, timeout=15)
    print("malformed: FAIL expected 400, got 200")
    failures += 1
except urllib.error.HTTPError as e:
    raw = e.read().decode()
    if e.code != 400:
        print(f"malformed: FAIL status {e.code} != 400")
        failures += 1
    elif "traceback" in raw.lower() or "GEMINI_API_KEY" in raw:
        print("malformed: FAIL leaks trace/secret")
        failures += 1
    elif "detail" not in raw:
        print("malformed: FAIL no detail field")
        failures += 1
    else:
        print("malformed: PASS 400 clean {detail}")
except Exception as e:
    print(f"malformed: FAIL {e!r}")
    failures += 1

# --- local no-key fallback (provider down still 200-equivalent path) ---
try:
    sys.path.insert(0, ".")
    import config as _cfg

    _saved = list(_cfg.GEMINI_API_KEYS)
    _saved1 = (_cfg.GEMINI_API_KEY, _cfg.GEMINI_API_KEY_2, _cfg.GEMINI_API_KEY_3)
    _cfg.GEMINI_API_KEYS = []
    _cfg.GEMINI_API_KEY = _cfg.GEMINI_API_KEY_2 = _cfg.GEMINI_API_KEY_3 = None
    from llm_interpreter import interpret_notes as _interp

    out = _interp(["Solar drop 1-3PM, 80% reduction."], 200.0, timeout_s=5.0)
    assert len(out) == 1, out
    # Provider down: safe no_op OR deterministic fallback salvage
    # (explanation carries the 'fallback:' prefix). Never raises.
    assert out[0].directive_type in (
        "no_op",
        "solar_reduction",
        "no_charge_window",
        "no_discharge_window",
        "minimum_battery_reserve",
        "max_grid_window",
    ), out
    if out[0].directive_type != "no_op":
        assert out[0].explanation.startswith("fallback:"), out
    print(f"no-key fallback: PASS {out[0].directive_type}, never raises")
    _cfg.GEMINI_API_KEYS = _saved
    (_cfg.GEMINI_API_KEY, _cfg.GEMINI_API_KEY_2, _cfg.GEMINI_API_KEY_3) = _saved1
except Exception as e:
    print(f"no-key fallback: FAIL {e!r}")
    failures += 1

print(f"\n{'GATE PASSED' if not failures else 'GATE FAILED'} ({failures} failures)")
sys.exit(1 if failures else 0)
