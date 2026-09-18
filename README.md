# GridWise — Smart Campus Energy Optimization API

BUP CSE Fest 2026 · Online Preliminary Round.

GridWise takes a 24-hour campus energy scenario (hourly demand, rooftop
solar availability, grid tariff, battery specs) plus 1–3 free-text operator
notes, and returns two things: a machine-checkable interpretation of each
note, and a 24-hour operating schedule that obeys every applicable
directive while minimizing grid electricity cost.

## Request / response contract

`POST /optimize-energy` accepts one scenario object:

- `scenario_id` — echoed back verbatim.
- `operator_notes` — 1–3 non-empty strings.
- `hours` — exactly 24 entries, one per hour 0–23, each with
  `demand_kwh`, `solar_kwh`, `tariff_bdt_per_kwh`.
- `battery` — `capacity_kwh`, `initial_energy_kwh`,
  `minimum_energy_kwh`, `max_charge_kwh_per_hour`,
  `max_discharge_kwh_per_hour`.

It returns:

- `directive_interpretation` — exactly one entry per note, in
  `note_index` order, each with `applies`, `directive_type`,
  `structured_adjustment`, and a short `explanation`.
  Irrelevant notes come back as
  `applies: false, directive_type: no_op, structured_adjustment: null`.
- `hourly_plan` — 24 entries with `grid_kwh`, `solar_used_kwh`,
  `battery_action` (`charge`/`discharge`/`idle`), `battery_kwh`, and
  `battery_energy_after_kwh`.
- `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` — recomputed from
  `hourly_plan`, which is the source of truth.
- `plan_summary` — short human-readable description of the strategy.
  Wording is free; nothing scores it.

`GET /health` returns `{"status": "ok"}` and depends on nothing else —
no key, no model, no solver.

## Supported directives

| `directive_type` | Effect on the schedule |
|---|---|
| `solar_reduction` | Usable solar in the listed hours is scaled by `factor`, the fraction remaining (an 80% reduction means `factor: 0.2`). |
| `minimum_battery_reserve` | Stored energy after each listed hour must stay at or above `minimum_energy_kwh`. Percentage reserves in notes are converted to kWh using the request's battery capacity. |
| `no_charge_window` | Battery charging is zero in the listed hours. |
| `no_discharge_window` | Battery discharging is zero in the listed hours. |
| `max_grid_window` | Grid import in each listed hour is capped at `max_grid_kwh`. |
| `no_op` | The note does not affect the schedule. Always `applies: false` with a null adjustment. |

Time windows use whole-hour intervals, start-inclusive end-exclusive:
1 PM to 3 PM means hours `[13, 14]`.

## How it works

```
request → schemas → llm_interpreter → guardrails → optimizer → response
main.py   schemas.py  llm_interpreter.py  guardrails.py  optimizer.py
```

**LLM (`llm_interpreter.py`).** Google Gemini `gemini-3.5-flash-lite`
through the `google-genai` SDK. One batched call covers all notes in the
request (temperature 0, JSON-only response). It translates English into
raw directive JSON and does no energy math. Requests start on the next
key in the pool (round-robin) and fail over to the next key on
quota, overload, auth, timeout, or malformed-output failures, inside a
single ~15 second budget. Per-note results are cached by note text and
battery capacity, so repeated notes answer without a model call. If the
model's output is rejected by validation on a note that clearly
mentions energy equipment, one corrective call is made with the
rejection reason attached. If every key fails, a small pattern matcher
covers only the obvious cases (explicit charge/discharge bans,
percentage solar drops, plain distractors); anything else degrades to a
safe default. Nothing the model returns is trusted until it passes
guardrails.

**Guardrails (`guardrails.py`).** Pure function, no I/O, and the only
place allowed to accept or reject model output. It enforces the
directive vocabulary, one entry per note in order, `no_op` semantics,
hours as unique integers 0–23 in ascending order, solar factor within
`[0, 1]`, reserves within `[0, capacity]`, and finite non-negative grid
caps. Anything invalid becomes `no_op`. It never raises and never
crashes the service.

**Optimizer (`optimizer.py`).** Pure PuLP linear program, no I/O. It
minimizes total grid cost Σ(grid × tariff) subject to hourly energy
balance, effective solar after reductions, battery bounds and
charge/discharge rate limits, state transitions, directive
windows/caps/reserves, and end-of-day neutrality (final stored energy
equals the starting level). A time-limited relaxed variant with a
penalized slack keeps adversarial grid-cap combinations from failing
hard. Rounded outputs re-derive grid from the balance equation so the
reported totals match the plan exactly.

**Routes (`main.py`).** FastAPI wiring only. The whole
interpret-validate-optimize pipeline runs in a worker thread under a
28-second deadline so a slow model call cannot block the event loop or
breach the 30-second per-request limit. Any failure degrades to a valid
schedule with all notes marked `no_op`. Malformed JSON returns 400,
schema violations return 422, and unexpected errors return a clean 500
with no stack traces and no secret values.

**Config (`config.py`).** The only module that reads environment
variables: `GEMINI_API_KEY` through `GEMINI_API_KEY_5`,
`GEMINI_API_KEYS` (optional comma-separated extra keys), and
`GEMINI_MODEL`. Missing keys never break import or `/health`; they
surface as controlled fallbacks in the interpretation path.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env       # Windows (cp .env.example .env on Linux/macOS)
```

Then put real keys in `.env`. That file is gitignored and must never be
committed. `.env.example` holds variable names only.

| Variable | Meaning | Required |
|---|---|---|
| `GEMINI_API_KEY` … `GEMINI_API_KEY_5` | Gemini API keys, primary plus pool keys for parallel requests and failover | First key required, rest optional |
| `GEMINI_API_KEYS` | Optional comma-separated extra keys, appended to the pool | No |
| `GEMINI_MODEL` | Model id, defaults to `gemini-3.5-flash-lite` | No |

Each teammate should use a personal key for local development. The key
configured on the deployed service is the one that counts during judging.

## Run

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Health check (Windows uses `curl.exe` to bypass the PowerShell alias):

```bash
curl.exe http://127.0.0.1:8000/health
# {"status":"ok"}
```

To build a sample request from the public pack:

```bash
python -c "import json; p=json.load(open('BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json')); json.dump(p['cases'][0]['input'], open('sample_request.json','w'), indent=2)"
curl.exe -X POST http://127.0.0.1:8000/optimize-energy -H "Content-Type: application/json" --data "@sample_request.json"
```

## Testing

Start the server first, then from a second shell in the repo root:

```bash
python tests/test_samples.py
# optional: python tests/test_samples.py --base-url http://127.0.0.1:8000
```

This replays all 10 public cases: interpretation semantics against the
reference pack, plan validity against ground-truth directives (energy
balance, solar limits, battery bounds and rates, directive windows and
caps, end-of-day neutrality), and totals recomputed from `hourly_plan`.
Expect `10/10 passed`.

```bash
python tests/test_paraphrase.py   # same directive, different wordings
python tests/test_load.py         # repeated-request stability and latency
```

## Docker

```bash
docker build -t gridwise:latest .
docker run -p 8000:8000 -e GEMINI_API_KEY=... gridwise:latest
curl.exe http://localhost:8000/health
```

The image serves on `0.0.0.0:8000`, includes a container health check on
`/health`, and bakes in no secrets — keys are passed at runtime with
`-e`. Production runs two uvicorn workers; local development uses one.

## Failure behavior

| Input | Result |
|---|---|
| Malformed JSON body | `400` with the validation detail, no trace |
| Well-formed but invalid (wrong hour count, 0 or 4+ notes, empty note) | `422` with the validation detail |
| Model timeout, quota error, bad JSON, missing key | `200` with affected notes as `no_op` and a valid schedule |
| Pipeline over 28 seconds | `200` with the same controlled fallback |
| Anything unexpected | `500 {"detail": "Internal server error."}` |

## Known limitations

- Percentage reserves are converted with the request's battery capacity.
  A note stating neither a percent nor a kWh value cannot produce a
  reserve and falls back to `no_op`.
- Clock times without AM/PM markers are resolved by context (solar
  language implies daytime). Genuinely ambiguous notes can mis-resolve.
- Valid judge scenarios are feasible by organizer guarantee. Hand-built
  infeasible combinations get the relaxed solver first, then a clamped
  fallback whose grid caps hold but whose energy balance may not — such
  inputs are outside the scored contract.

## Layout

```
main.py            # routes only
schemas.py         # request/response models, single source of truth
llm_interpreter.py # Gemini translation, key pool, cache, retries
guardrails.py      # deterministic validation of model output
optimizer.py       # PuLP linear program, pure function
config.py          # environment variables, nothing else reads them
tests/             # public-sample, paraphrase, and load checks
Dockerfile         # 0.0.0.0:8000, healthcheck, no baked secrets
```

FastAPI · Pydantic v2 · google-genai (Gemini) · PuLP (CBC) · uvicorn ·
python-dotenv.
