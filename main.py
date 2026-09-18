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
    )
    directives: List[DirectiveInterpretation] = validate_directives(
        raw, len(req.operator_notes), req.battery
    )
    plan_dicts = optimize(req.hours, req.battery, directives)
    return directives, plan_dicts


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
        plan_dicts = optimize(req.hours, req.battery, directives)

    hours_sorted = sorted(req.hours, key=lambda h: h.hour)
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours_sorted}
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
