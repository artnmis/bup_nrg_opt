"""GridWise — Smart Campus Energy Optimization API.

Routes only. Pipeline: request -> schemas -> llm_interpreter ->
guardrails -> optimizer -> response. No business logic here.
"""

from typing import List

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from guardrails import validate_directives
from optimizer import build_summary, optimize
from schemas import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeRequest,
    OptimizeResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-import the heavy genai SDK off the request path so the first POST
    # doesn't pay ~1s of import latency. Needs no key; /health stays
    # dependency-free regardless.
    try:
        await asyncio.to_thread(__import__, "google.genai")
    except Exception:
        pass
    yield


app = FastAPI(title="GridWise Optimize Energy API", lifespan=lifespan)

# B1: whole LLM+LP pipeline must finish inside the judge's 30s POST budget.
PIPELINE_TIMEOUT_S = 28.0
# #1: LLM (initial + capped corrective retry) owns at most 20s; LP owns at
# most ~6s (2 x 3s CBC caps); ~2s margin for guardrails/overhead.
LLM_TIMEOUT_S = 20.0


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _run_pipeline(req: OptimizeRequest) -> tuple:
    """Blocking LLM+guardrails+LP body. Runs in a worker thread so the
    event loop never blocks (B1). Lazy LLM import keeps /health key-free."""
    from llm_interpreter import interpret_notes

    raw = interpret_notes(
        list(req.operator_notes),
        float(req.battery.capacity_kwh),
        timeout_s=LLM_TIMEOUT_S,
    )
    directives: List[DirectiveInterpretation] = validate_directives(
        raw, len(req.operator_notes), req.battery
    )
    plan_dicts = optimize(req.hours, req.battery, directives)
    return directives, plan_dicts


def _fast_idle_plan(req: OptimizeRequest) -> list:
    """Solver-free fallback plan (#2): no LP, no I/O, runs in microseconds
    on the event loop without head-of-line-blocking /health or concurrent
    samples. All notes are no_op so effective solar == forecast."""
    hmap = {h.hour: h for h in req.hours}
    e0 = float(req.battery.initial_energy_kwh)
    plan = []
    for h in sorted(hmap.keys()):
        entry = hmap[h]
        demand = float(entry.demand_kwh)
        solar = float(entry.solar_kwh)
        s = round(min(solar, demand), 4)
        g = round(demand - s, 4)
        plan.append(
            {
                "hour": h,
                "grid_kwh": max(g, 0.0),
                "solar_used_kwh": max(s, 0.0),
                "battery_action": "idle",
                "battery_kwh": 0.0,
                "battery_energy_after_kwh": round(e0, 4),
            }
        )
    return plan


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(req: OptimizeRequest) -> OptimizeResponse:
    try:
        directives, plan_dicts = await asyncio.wait_for(
            asyncio.to_thread(_run_pipeline, req), timeout=PIPELINE_TIMEOUT_S
        )
    except Exception:
        # Controlled degradation: valid idle schedule, all notes no_op.
        # Covers LLM failure, LP failure, AND pipeline timeout.
        # Never a crash, never a stack trace (rules.md §5/§7).
        # #2: solver-free idle — never blocks the event loop with CBC.
        directives = [
            DirectiveInterpretation(
                note_index=i,
                applies=False,
                directive_type="no_op",
                structured_adjustment=None,
                explanation="Internal processing fallback; defaulted to no_op.",
            )
            for i in range(len(req.operator_notes))
        ]
        plan_dicts = _fast_idle_plan(req)

    hours_sorted = sorted(req.hours, key=lambda h: h.hour)
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours_sorted}
    try:
        hourly_plan = [HourlyPlanEntry(**p) for p in plan_dicts]
    except Exception:
        # Response-side guard (#12): never emit a malformed plan.
        plan_dicts = _fast_idle_plan(req)
        hourly_plan = [HourlyPlanEntry(**p) for p in plan_dicts]
    # Response self-check (#12): 24 entries, non-negative, totals agree
    # (totals are recomputed below, so agreement is by construction; the
    # checks below guard against negative/solver-dust regressions).
    if len(hourly_plan) != 24 or sorted(p.hour for p in hourly_plan) != list(range(24)):
        plan_dicts = _fast_idle_plan(req)
        hourly_plan = [HourlyPlanEntry(**p) for p in plan_dicts]
    elif any(p.grid_kwh < -0.011 or p.solar_used_kwh < -0.011 or p.battery_kwh < -0.011 for p in hourly_plan):
        plan_dicts = _fast_idle_plan(req)
        hourly_plan = [HourlyPlanEntry(**p) for p in plan_dicts]

    total_grid = round(sum(p.grid_kwh for p in hourly_plan), 4)
    total_cost = round(sum(p.grid_kwh * tariff[p.hour] for p in hourly_plan), 4)
    peak_grid = round(max(p.grid_kwh for p in hourly_plan), 4)

    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=build_summary(directives),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors and all(e.get("type") == "json_invalid" for e in errors):
        return JSONResponse(status_code=400, content={"detail": errors})
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if request.url.path == "/health":
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
