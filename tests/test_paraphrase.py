"""Paraphrase robustness: same directive, any wording -> same JSON.

Usage (server must already be running):
    python tests/test_paraphrase.py [--base-url http://127.0.0.1:8000]

Checks (all via POST /optimize-energy, tolerance 0.01):
- 4 solar wordings -> solar_reduction [13,14] factor 0.2
- 50% of 200 kWh reserve -> minimum_battery_reserve [18,19,20] 100
- don't-charge 2-4PM -> no_charge_window [14,15]
"""

import json
import sys
import urllib.request

BASE_URL = sys.argv[sys.argv.index("--base-url") + 1] if "--base-url" in sys.argv else "http://127.0.0.1:8000"
TOL = 0.011


def make_request(note, capacity=200.0):
    hours = [
        {"hour": h, "demand_kwh": 100.0, "solar_kwh": 50.0, "tariff_bdt_per_kwh": 10.0}
        for h in range(24)
    ]
    return {
        "scenario_id": "PARA-TEST",
        "operator_notes": [note],
        "hours": hours,
        "battery": {
            "capacity_kwh": capacity,
            "initial_energy_kwh": 100.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 50.0,
            "max_discharge_kwh_per_hour": 50.0,
        },
    }


def post(payload):
    req = urllib.request.Request(
        f"{BASE_URL}/optimize-energy",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


CHECKS = [
    {
        "name": "solar 24h-format",
        "note": "PV production will drop to about 20% between 13:00 and 15:00.",
        "want_type": "solar_reduction",
        "want_adj": {"hours": [13, 14], "factor": 0.2},
    },
    {
        "name": "solar ambiguous words",
        "note": "Panel washing from one until three will leave roughly one-fifth of normal solar output.",
        "want_type": "solar_reduction",
        "want_adj": {"hours": [13, 14], "factor": 0.2},
    },
    {
        "name": "solar 80pct reduction",
        "note": "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.",
        "want_type": "solar_reduction",
        "want_adj": {"hours": [13, 14], "factor": 0.2},
    },
    {
        "name": "solar plain drop-to",
        "note": "Solar output will drop to about 20% from 1 PM to 3 PM.",
        "want_type": "solar_reduction",
        "want_adj": {"hours": [13, 14], "factor": 0.2},
    },
    {
        "name": "reserve pct->kWh",
        "note": "Keep at least 50% of the battery capacity stored in the battery from 6 PM until 9 PM for emergency operations.",
        "want_type": "minimum_battery_reserve",
        "want_adj": {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0},
        "capacity": 200.0,
    },
    {
        "name": "no-charge window",
        "note": "Do not charge the battery between 2 PM and 4 PM.",
        "want_type": "no_charge_window",
        "want_adj": {"hours": [14, 15]},
    },
]


def adj_close(got, want):
    if set(got.keys()) != set(want.keys()):
        return False
    for k, v in want.items():
        g = got.get(k)
        if isinstance(v, list):
            if g != v:
                return False
        else:
            try:
                if abs(float(g) - float(v)) > TOL:
                    return False
            except Exception:
                return False
    return True


failures = 0
for c in CHECKS:
    try:
        resp = post(make_request(c["note"], c.get("capacity", 200.0)))
    except Exception as e:
        print(f"{c['name']}: FAIL http error: {e}")
        failures += 1
        continue
    di = resp.get("directive_interpretation", [])
    if len(di) != 1:
        print(f"{c['name']}: FAIL directive count {len(di)} != 1")
        failures += 1
        continue
    g = di[0]
    if g["directive_type"] != c["want_type"] or g["applies"] is not True:
        print(f"{c['name']}: FAIL got {g['directive_type']}/{g['applies']}, want {c['want_type']}/True")
        failures += 1
    elif not isinstance(g["structured_adjustment"], dict) or not adj_close(g["structured_adjustment"], c["want_adj"]):
        print(f"{c['name']}: FAIL adjustment {g['structured_adjustment']} != {c['want_adj']}")
        failures += 1
    else:
        print(f"{c['name']}: PASS")

print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
sys.exit(1 if failures else 0)
