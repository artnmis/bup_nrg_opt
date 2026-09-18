"""Runs the 10 public sample cases against a running local API.

Usage (server must already be running):
    python -m uvicorn main:app --host 127.0.0.1 --port 8000
    python tests/test_samples.py [--base-url http://127.0.0.1:8000]

Checks per case: directive interpretation semantics (type/applies/hours/
numbers — NOT plan_summary wording), 24-hour plan shape, energy balance,
battery bounds/rate limits, directive application, end-of-day neutrality,
and totals recalculated from hourly_plan. Tolerance 0.01.
"""

import json
import sys
import urllib.request

BASE_URL = sys.argv[sys.argv.index("--base-url") + 1] if "--base-url" in sys.argv else "http://127.0.0.1:8000"
TOL = 0.011

pack = json.load(open("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"))
failures = 0

for case in pack["cases"]:
    cid = case["id"]
    inp = case["input"]
    exp = case["expected_output"]
    problems = []

    def post(payload):
        req = urllib.request.Request(
            f"{BASE_URL}/optimize-energy",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.load(urllib.request.urlopen(req, timeout=60))

    try:
        resp = post(inp)
    except Exception as e:
        print(f"{cid}: FAIL http error: {e}")
        failures += 1
        continue

    if resp.get("scenario_id") != inp["scenario_id"]:
        problems.append("scenario_id echo mismatch")

    # --- interpretation semantics ---
    got_di = resp.get("directive_interpretation", [])
    exp_di = exp["directive_interpretation"]
    if len(got_di) != len(exp_di):
        problems.append(f"directive count {len(got_di)} != {len(exp_di)}")
    else:
        for g, e in zip(got_di, exp_di):
            if g["note_index"] != e["note_index"]:
                problems.append(f"note {e['note_index']}: index order wrong")
            if g["directive_type"] != e["directive_type"] or g["applies"] != e["applies"]:
                problems.append(f"note {e['note_index']}: got {g['directive_type']}/{g['applies']}, want {e['directive_type']}/{e['applies']}")
            elif g["structured_adjustment"] != e["structured_adjustment"]:
                problems.append(f"note {e['note_index']}: adjustment {g['structured_adjustment']} != {e['structured_adjustment']}")

    # --- plan validity, replayed against GROUND-TRUTH (expected) directives ---
    hmap = {h["hour"]: h for h in inp["hours"]}
    bat = inp["battery"]
    eff_solar = {h: hmap[h]["solar_kwh"] for h in range(24)}
    reserve = {h: bat["minimum_energy_kwh"] for h in range(24)}
    no_ch, no_dis, caps = set(), set(), {}
    for e in exp_di:
        adj = e["structured_adjustment"] or {}
        t = e["directive_type"]
        if t == "solar_reduction":
            for h in adj["hours"]:
                eff_solar[h] *= adj["factor"]
        elif t == "minimum_battery_reserve":
            for h in adj["hours"]:
                reserve[h] = max(reserve[h], adj["minimum_energy_kwh"])
        elif t == "no_charge_window":
            no_ch.update(adj["hours"])
        elif t == "no_discharge_window":
            no_dis.update(adj["hours"])
        elif t == "max_grid_window":
            for h in adj["hours"]:
                caps[h] = min(caps.get(h, float("inf")), adj["max_grid_kwh"])

    plan = resp.get("hourly_plan", [])
    if sorted(p["hour"] for p in plan) != list(range(24)):
        problems.append("hourly_plan must cover 0..23 exactly")
    else:
        e_prev = bat["initial_energy_kwh"]
        for p in plan:
            h = p["hour"]
            ch = p["battery_kwh"] if p["battery_action"] == "charge" else 0.0
            dis = p["battery_kwh"] if p["battery_action"] == "discharge" else 0.0
            if p["battery_action"] == "idle" and abs(p["battery_kwh"]) > TOL:
                problems.append(f"h{h}: idle with nonzero battery_kwh")
            if abs(p["grid_kwh"] + p["solar_used_kwh"] + dis - hmap[h]["demand_kwh"] - ch) > TOL:
                problems.append(f"h{h}: energy balance violated")
            if p["solar_used_kwh"] - eff_solar[h] > TOL:
                problems.append(f"h{h}: solar overuse")
            if p["battery_energy_after_kwh"] < reserve[h] - TOL or p["battery_energy_after_kwh"] > bat["capacity_kwh"] + TOL:
                problems.append(f"h{h}: battery bound violated")
            if ch - bat["max_charge_kwh_per_hour"] > TOL or dis - bat["max_discharge_kwh_per_hour"] > TOL:
                problems.append(f"h{h}: rate limit violated")
            if h in no_ch and ch > TOL:
                problems.append(f"h{h}: charged during no_charge_window")
            if h in no_dis and dis > TOL:
                problems.append(f"h{h}: discharged during no_discharge_window")
            if h in caps and p["grid_kwh"] - caps[h] > TOL:
                problems.append(f"h{h}: grid cap violated")
            if abs(e_prev + ch - dis - p["battery_energy_after_kwh"]) > TOL:
                problems.append(f"h{h}: battery transition violated")
            e_prev = p["battery_energy_after_kwh"]
        if abs(e_prev - bat["initial_energy_kwh"]) > TOL:
            problems.append("end-of-day neutrality violated")

        tg = sum(p["grid_kwh"] for p in plan)
        tc = sum(p["grid_kwh"] * hmap[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
        pk = max(p["grid_kwh"] for p in plan)
        if abs(tg - resp["total_grid_kwh"]) > TOL:
            problems.append("total_grid_kwh mismatch")
        if abs(tc - resp["total_cost_bdt"]) > TOL:
            problems.append("total_cost_bdt mismatch")
        if abs(pk - resp["peak_grid_kwh"]) > TOL:
            problems.append("peak_grid_kwh mismatch")

    if problems:
        failures += 1
        print(f"{cid}: FAIL")
        for pr in problems:
            print(f"    - {pr}")
    else:
        print(f"{cid}: PASS (cost {resp['total_cost_bdt']}, ref {exp['total_cost_bdt']})")

print(f"\n{len(pack['cases']) - failures}/{len(pack['cases'])} passed")
sys.exit(1 if failures else 0)
