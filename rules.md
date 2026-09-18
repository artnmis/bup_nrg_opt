# Project Rules — GridWise (BUP CSE Fest 2026)

These are the rules everyone on the team follows. Read this before writing code.

## 1. Stack (locked — don't swap mid-event)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Web framework | FastAPI |
| Validation | Pydantic v2 |
| LLM | Google Gemini — **`gemini-3.5-flash-lite`** |
| LLM SDK | `google-genai` (the current Google Gen AI SDK) |
| Optimizer | PuLP (linear programming) |
| Server | uvicorn |
| Containerization | Docker |
| Deployment | Render / Railway / Fly.io (whichever gets a public URL first — no login wall) |

Do not introduce a different LLM provider, a different web framework, or a different language for any part of the service without agreeing with the team first — the judge only cares about one deployed service with two exact endpoints, so fragmenting the stack costs time, not points.

## 2. Repo structure

```
/app
  main.py            # FastAPI app + routes only. No business logic here.
  schemas.py         # Pydantic request/response models — the single source of truth
                      # for every field name and enum. Import from here, never
                      # hand-write a dict key that duplicates the contract.
  llm_interpreter.py # Calls Gemini, returns raw directive JSON per operator note.
  guardrails.py      # Validates/repairs/rejects the LLM's output before the
                      # optimizer ever sees it. Nothing from the LLM reaches
                      # optimizer.py unvalidated.
  optimizer.py        # Builds and solves the LP. Pure function: (hours, battery,
                      # validated directives) -> hourly_plan. No I/O.
  config.py           # Loads env vars (API keys, model name). Nothing else reads
                      # os.environ directly.
/tests
  test_samples.py     # Runs the 10 public sample cases against the running API
                      # and checks the response against expected fields.
Dockerfile
requirements.txt
.env.example          # Variable NAMES only, no real values. Committed.
.env                   # Real values. Gitignored, never committed.
README.md
rules.md               # this file
prerequisite.md
```

If you need a new file, it goes in `/app` and gets a `snake_case.py` name that matches its single responsibility. Don't dump unrelated logic into `main.py`.

## 3. Naming conventions

- **Python**: `snake_case` for functions/variables, `PascalCase` for Pydantic models and classes, `UPPER_SNAKE` for constants.
- **Files**: `snake_case.py`.
- **Env vars**: `UPPER_SNAKE`, prefixed by concern — see `prerequisite.md` for the exact names. Don't invent new ones without updating `.env.example`.
- **Branches**: `feature/optimizer`, `feature/llm-interpreter`, `fix/battery-bounds`.
- **Commits**: short, imperative, describes what changed — `add battery reserve constraint`, not `fixed stuff` or `wip`.
- **API field names**: must match the Problem Statement exactly (`scenario_id`, `operator_notes`, `directive_interpretation`, `hourly_plan`, `battery_action`, etc.). These are not up for stylistic changes — a renamed field is a broken contract and an automatic zero on that check.

## 4. The pipeline contract (do not collapse these stages)

```
request → schema validation → LLM interpretation → guardrails → optimizer → response
```

- The LLM is only allowed to produce a `directive_interpretation` entry per note. It never touches the optimization math directly.
- `guardrails.py` is the only thing allowed to accept or reject LLM output. If the LLM returns something malformed or an unsupported `directive_type`, guardrails catches it — the service must not crash and must not silently invent a new directive type.
- `optimizer.py` only ever receives already-validated directives. It has no idea an LLM exists.
- Every non-`no_op` directive: `applies = true`. `no_op` is the only directive allowed `applies = false`, and it must have `structured_adjustment = null`.
- Supported `directive_type` values, full stop — nothing else is emitted: `solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`, `max_grid_window`, `no_op`.

## 5. Testing discipline

- Before opening a PR that touches `/optimize-energy`, run all 10 public sample cases against your local server and confirm the response shape and directive interpretations look right (`plan_summary` wording doesn't need to match, everything else should be structurally correct).
- `GET /health` must always return `200 {"status": "ok"}` — if a change breaks this, it doesn't get merged, full stop.
- Never let a malformed request or an LLM/provider failure return a 500 with a stack trace or any secret value in the body. Catch it, return a clean error.

## 6. Git workflow

- Create a new branch per feature (`feature/xxx`), don't commit straight to `main`.
- Small, frequent commits over one giant commit at the end.
- Before merging: pull `main`, resolve conflicts locally, make sure `/health` and one sample case still pass.
- Whoever touches `schemas.py` pings the team in chat — it's the shared contract, changes there ripple everywhere.

## 7. Secrets & security

- Never commit `.env`, API keys, tokens, or passwords. `.env` is gitignored from commit 1.
- `.env.example` lists variable **names** only.
- Don't log full LLM prompts/responses if they could ever contain a key or token — logs are visible to whoever inspects the deployment.
- Only use the synthetic data given by the challenge — no real campus/utility/personal data, ever.
- Errors returned to the client (400/422/500) never include stack traces or internal exception text.

## 8. Repository visibility

- Create the repo fresh after question reveal (not before).
- Keep it **private** during the event.
- Make it **public** only after the submission deadline.

## 9. README ownership

Whoever finishes deployment + Docker last is responsible for making sure `README.md` has: setup steps, exact run command, env var names (not values), model/provider used (`gemini-3.5-flash-lite`), optimizer/library used (PuLP), a sample curl for `/health` and `/optimize-energy`, the public-sample test command, dependencies, and known limitations. This gets checked for reproducibility from a clean environment — test it yourself before submitting, don't assume it works.