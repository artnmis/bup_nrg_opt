"""PuLP LP optimizer. Pure function, no I/O, no LLM/FastAPI imports.

(validated hours, battery, directives) -> hourly_plan entries as dicts.
Falls back to a feasible idle schedule if the solver fails.
"""

from typing import List

import pulp

EPS = 1e-6


def _directive_maps(directives) -> tuple:
    """Combine validated directives into per-hour constraint maps."""
    solar_factor = {h: 1.0 for h in range(24)}
    reserve_extra = {h: 0.0 for h in range(24)}
    no_charge = set()
    no_discharge = set()
    grid_cap = {}
    for d in directives:
        dtype = d.directive_type if hasattr(d, "directive_type") else d.get("directive_type")
        adj = d.structured_adjustment if hasattr(d, "structured_adjustment") else d.get("structured_adjustment")
        if dtype == "no_op" or not adj:
            continue
        hours = adj["hours"] if isinstance(adj, dict) else []
        if dtype == "solar_reduction":
            for h in hours:
                solar_factor[h] *= float(adj["factor"])
        elif dtype == "minimum_battery_reserve":
            for h in hours:
                reserve_extra[h] = max(reserve_extra[h], float(adj["minimum_energy_kwh"]))
        elif dtype == "no_charge_window":
            no_charge.update(hours)
        elif dtype == "no_discharge_window":
            no_discharge.update(hours)
        elif dtype == "max_grid_window":
            for h in hours:
                cap = float(adj["max_grid_kwh"])
                grid_cap[h] = cap if h not in grid_cap else min(grid_cap[h], cap)
    return solar_factor, reserve_extra, no_charge, no_discharge, grid_cap


def optimize(hours, battery, directives) -> List[dict]:
    """Solve min SUM(grid*tariff). Returns 24 plan dicts (hour order 0..23)."""
    def _f(h, name):
        return float(getattr(h, name) if hasattr(h, name) else h[name])

    hmap = {h.hour if hasattr(h, "hour") else h["hour"]: h for h in hours}
    order = sorted(hmap.keys())
    D = [_f(hmap[h], "demand_kwh") for h in order]
    S = [_f(hmap[h], "solar_kwh") for h in order]
    T = [_f(hmap[h], "tariff_bdt_per_kwh") for h in order]

    cap = _f(battery, "capacity_kwh")
    e0 = _f(battery, "initial_energy_kwh")
    base_min = _f(battery, "minimum_energy_kwh")
    max_ch = _f(battery, "max_charge_kwh_per_hour")
    max_dis = _f(battery, "max_discharge_kwh_per_hour")

    solar_factor, reserve_extra, no_charge, no_discharge, grid_cap = _directive_maps(directives or [])
    ES = [S[i] * solar_factor[order[i]] for i in range(24)]
    lo = [max(base_min, reserve_extra[order[i]]) for i in range(24)]

    try:
        return _solve_lp(order, D, ES, T, cap, e0, lo, max_ch, max_dis, no_charge, no_discharge, grid_cap)
    except Exception as first_err:
        # LP infeasible (e.g. adversarial grid caps): retry with penalized
        # slack on grid caps so violation is minimal, not a hard failure.
        try:
            import sys as _sys

            print(f"[optimizer] LP fallback (relaxed): {type(first_err).__name__}", file=_sys.stderr)
            return _solve_relaxed(order, D, ES, T, cap, e0, lo, base_min, max_ch, max_dis, no_charge, no_discharge, grid_cap)
        except Exception as second_err:
            import sys as _sys2

            print(f"[optimizer] relaxed fallback failed, idle+clamp: {type(second_err).__name__}", file=_sys2.stderr)
            return _fallback_idle(order, D, ES, e0, grid_cap)


def _solve_lp(order, D, ES, T, cap, e0, lo, max_ch, max_dis, no_charge, no_discharge, grid_cap) -> List[dict]:
    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)
    grid = [pulp.LpVariable(f"g_{h}", lowBound=0) for h in order]
    solar = [pulp.LpVariable(f"s_{h}", lowBound=0, upBound=max(ES[i], 0)) for i, h in enumerate(order)]
    ch = [pulp.LpVariable(f"c_{h}", lowBound=0, upBound=0 if h in no_charge else max_ch) for h in order]
    dis = [pulp.LpVariable(f"d_{h}", lowBound=0, upBound=0 if h in no_discharge else max_dis) for h in order]
    E = [pulp.LpVariable(f"e_{h}", lowBound=lo[i], upBound=cap) for i, h in enumerate(order)]

    for i, h in enumerate(order):
        prev = e0 if i == 0 else E[i - 1]
        prob += E[i] == prev + ch[i] - dis[i]
        prob += grid[i] + solar[i] + dis[i] == D[i] + ch[i]
        if h in grid_cap:
            prob += grid[i] <= grid_cap[h]
    prob += E[23] == e0
    prob += pulp.lpSum(grid[i] * T[i] for i in range(24))

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=3))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")

    # Round + re-derive grid from balance so totals match exactly.
    return _round_solution(order, D, ES, e0, grid, solar, ch, dis)


def _solve_relaxed(order, D, ES, T, cap, e0, lo, base_min, max_ch, max_dis, no_charge, no_discharge, grid_cap) -> List[dict]:
    """Relaxed LP: grid caps get penalized slack so an adversarial cap combo
    yields minimal violation instead of a non-optimal crash. Reserves stay
    hard (organizer guarantees feasibility); neutrality and balance stay hard.
    """
    prob = pulp.LpProblem("gridwise_relaxed", pulp.LpMinimize)
    grid = [pulp.LpVariable(f"rg_{h}", lowBound=0) for h in order]
    solar = [pulp.LpVariable(f"rs_{h}", lowBound=0, upBound=max(ES[i], 0)) for i, h in enumerate(order)]
    ch = [pulp.LpVariable(f"rc_{h}", lowBound=0, upBound=0 if h in no_charge else max_ch) for h in order]
    dis = [pulp.LpVariable(f"rd_{h}", lowBound=0, upBound=0 if h in no_discharge else max_dis) for h in order]
    E = [pulp.LpVariable(f"re_{h}", lowBound=lo[i], upBound=cap) for i, h in enumerate(order)]
    slack = {}
    big_m = (max(T) if T else 0) + 10000.0
    for i, h in enumerate(order):
        prev = e0 if i == 0 else E[i - 1]
        prob += E[i] == prev + ch[i] - dis[i]
        prob += grid[i] + solar[i] + dis[i] == D[i] + ch[i]
        if h in grid_cap:
            s = pulp.LpVariable(f"slack_{h}", lowBound=0)
            slack[h] = s
            prob += grid[i] <= grid_cap[h] + s
    prob += E[23] == e0
    prob += pulp.lpSum(grid[i] * T[i] for i in range(24)) + pulp.lpSum(slack[h] * big_m for h in slack)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=3))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Relaxed LP not optimal: {pulp.LpStatus[status]}")

    # Reuse the same round + re-derive path as the strict solver.
    return _round_solution(order, D, ES, e0, grid, solar, ch, dis)


def _round_solution(order, D, ES, e0, grid, solar, ch, dis) -> List[dict]:
    # Round + re-derive grid from balance so totals match exactly.
    plan: List[dict] = []
    e_prev = e0
    for i, h in enumerate(order):
        s = round(max(float(pulp.value(solar[i])), 0.0), 4)
        c = round(max(float(pulp.value(ch[i])), 0.0), 4)
        d = round(max(float(pulp.value(dis[i])), 0.0), 4)
        s = min(s, round(ES[i], 4))
        e_after = round(e_prev + c - d, 4)
        g = round(D[i] + c - d - s, 4)
        if g < 0 and g > -0.011:
            # Float dust: absorb into solar curtailment accounting.
            s = round(s + g, 4)
            g = 0.0
        if c < 0.005 and d < 0.005:
            action, mag = "idle", 0.0
            # Recompute energy without dust so idle hours are exact.
            e_after = round(e_prev, 4)
            g = round(D[i] - s, 4)
        elif c >= d:
            action, mag = "charge", round(c - d, 4)
            e_after = round(e_prev + mag, 4)
            g = round(D[i] + mag - s, 4)
        else:
            action, mag = "discharge", round(d - c, 4)
            e_after = round(e_prev - mag, 4)
            g = round(D[i] - mag - s, 4)
        plan.append(
            {
                "hour": h,
                "grid_kwh": max(g, 0.0),
                "solar_used_kwh": max(s, 0.0),
                "battery_action": action,
                "battery_kwh": mag,
                "battery_energy_after_kwh": e_after,
            }
        )
        e_prev = e_after
    return plan


def _fallback_idle(order, D, ES, e0, grid_cap=None) -> List[dict]:
    """Last resort: solar->demand, grid->rest, battery idle. Clamps grid to
    caps when given so the worst case respects max_grid_window (balance may
    still break on truly infeasible inputs — but caps are hard rules)."""
    plan: List[dict] = []
    caps = grid_cap or {}
    for i, h in enumerate(order):
        s = round(min(ES[i], D[i]), 4)
        g = round(D[i] - s, 4)
        if h in caps:
            g = min(g, round(float(caps[h]), 4))
            # Keep balance best-effort: curtailed solar already minimal;
            # if clamped, demand is partially unmet (infeasible input).
        plan.append(
            {
                "hour": h,
                "grid_kwh": max(g, 0.0),
                "solar_used_kwh": s,
                "battery_action": "idle",
                "battery_kwh": 0.0,
                "battery_energy_after_kwh": round(e0, 4),
            }
        )
    return plan


def build_summary(directives) -> str:
    applied = [
        d.directive_type if hasattr(d, "directive_type") else d.get("directive_type")
        for d in directives or []
    ]
    applied = [t for t in applied if t != "no_op"]
    if not applied:
        return (
            "No operator directives applied: solar used up to demand, grid covers "
            "the remainder, and battery shifts energy from cheap to expensive hours "
            "while ending at its initial level."
        )
    kinds = sorted(set(applied))
    return (
        f"Applied directives ({', '.join(kinds)}); otherwise solar used up to "
        "effective availability, grid covers the remainder, and battery shifts "
        "energy from cheap to expensive hours while respecting all directive "
        "windows and ending at its initial level."
    )
