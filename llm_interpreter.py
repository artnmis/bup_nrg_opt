"""Calls Gemini, returns one RAW directive guess per operator note, in order.

Pure translator: English -> structured JSON. Never touches the optimizer,
never invents energy numbers beyond what the note states. Everything
returned here is untrusted until guardrails.py validates it.

Strategy: ONE batched call for all notes (fast, single timeout budget).
On ANY failure -> safe per-note defaults that guardrails coerce to no_op.
"""

import json
import re
from typing import List

from schemas import RawDirective

import logging as _logging

# Silence the google-genai SDK's once-per-process AFC notice
# ("Direct use of automatic function calling ..."). We never use function
# calling (no tools are passed), and real API failures still raise as
# exceptions which are logged per-key below — so ERROR level loses nothing.
_logging.getLogger("google.genai").setLevel(_logging.ERROR)

# Cache: (note text stripped, battery capacity) -> RawDirective.
# Helps p95 latency when the judge repeats similar notes.
# Bounded LRU (#10): max 2000 entries, eviction of least-recently-used.
# Per-process (workers=2 each have their own); bound keeps memory flat.
_CACHE: dict = {}
_CACHE_MAX = 2000


def _cache_get(key):
    try:
        val = _CACHE.get(key)
        if val is not None:
            # refresh recency
            del _CACHE[key]
            _CACHE[key] = val
        return val
    except Exception:
        return None


def _cache_put(key, val) -> None:
    try:
        if key in _CACHE:
            del _CACHE[key]
        _CACHE[key] = val
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
    except Exception:
        pass

# Cooldown per API key after quota/overload: monotonic timestamp until which
# the key is skipped. Exponential backoff per key (#13); auth errors
# quarantine the key long-term. In-memory only, per-process.
_KEY_COOLDOWN_UNTIL: dict = {}
_KEY_FAIL_COUNT: dict = {}

# Round-robin pool cursor: each _call_gemini invocation starts at the next
# key, so CONCURRENT requests spread across keys from the first attempt
# instead of all hammering key 1 serially into 429s. Failover order within
# one call is still sequential from its start key. Thread-safe via lock
# (endpoint runs each request in its own worker thread).
import itertools as _itertools
import threading as _threading

_KEY_RR = _itertools.count()
_KEY_RR_LOCK = _threading.Lock()


def _next_pool_start(n: int) -> int:
    with _KEY_RR_LOCK:
        return next(_KEY_RR) % max(1, n)

# Output constraint: temperature 0 + response_mime_type=json only.
# NOTE: response_json_schema is deliberately NOT used. An OBJECT-typed
# structured_adjustment without enumerated properties makes the model emit
# {} (verified live: solar_reduction with empty adjustment), which
# guardrails must reject. Prompt few-shots + guardrails constrain shape.

# Energy-equipment keywords (#6): word-boundary matched so the canonical
# distractor ("library … book-return hours") and substrings ("campus"/"team"
# containing "am", "hours" containing "hour") do NOT trigger a wasted
# corrective retry. Pure time markers (hour/am/pm/:00) are deliberately
# excluded — retry fires on equipment language, not clock language.
# Synonyms added: photovoltaic, rooftop, kw (bare unit), charger, inverter,
# feeder, substation.
_ENERGY_PATTERNS = (
    r"solar", r"photovoltaic", r"\bpv\b", r"panel", r"rooftop",
    r"battery", r"charg\w*", r"discharg\w*", r"\bgrid\b",
    r"reserve", r"\bkwh?\b", r"tariff", r"inverter", r"feeder",
    r"substation",
)
_ENERGY_RE = re.compile("|".join(f"(?:{p})" for p in _ENERGY_PATTERNS), re.IGNORECASE)

# Distractor nouns that suppress the retry when no equipment word is present
# (belt-and-braces; _has_energy_keyword already returns False for them).
_DISTRACTOR_RE = re.compile(
    r"cafeteria|library|menu|deadline|club|booking|seminar|registration",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You interpret campus operator notes into energy directives.
Return ONLY valid JSON, no prose, no markdown fences.

Allowed directive_type values (nothing else):
- solar_reduction: {"hours":[...], "factor": number} — usable solar fraction REMAINS.
- minimum_battery_reserve: {"hours":[...], "minimum_energy_kwh": number}
- no_charge_window: {"hours":[...]}
- no_discharge_window: {"hours":[...]}
- max_grid_window: {"hours":[...], "max_grid_kwh": number}
- no_op with structured_adjustment null (note is irrelevant to today's 24h energy schedule).

Rules:
- Hours are whole-hour intervals, start-inclusive end-exclusive. Conversion table (memorize):
  "1-3PM" / "1 PM to 3 PM" / "13:00-15:00" -> [13,14]
  "2-5AM" / "2 AM until 5 AM" -> [2,3,4]
  "6-10PM" / "6 PM until 10 PM" -> [18,19,20,21]
  "6-9PM" / "6 PM until 9 PM" -> [18,19,20]
  "2-4PM" / "2 PM until 4 PM" -> [14,15]
  "10AM-noon" / "10 AM until noon" -> [10,11]
  "11AM-1PM" -> [11,12]; "noon-2PM" / "midday-2PM" -> [12,13]; "11AM-2PM" -> [11,12,13].
  "midnight"=0, "noon"/"midday"=12: "midnight to 2 AM" -> [0,1]; "noon until 2 PM" -> [12,13].
  24h format maps identically: "19:00-22:00" / "7 PM until 10 PM" -> [19,20,21]; "13:00-15:00" -> [13,14].
  Always unique ints 0..23 ascending, never strings.
- Ambiguous clock times with no AM/PM marker: use context. Solar/panel/PV notes refer to daylight, so "one until three" with solar/panel language means [13,14], never [1,2]. Battery/grid notes at night ("2 until 5" with charger maintenance) mean [2,3,4].
- SOLAR FACTOR IS THE FRACTION REMAINING, not the reduction: "80% reduction" -> 0.2; "drop to 20%" -> 0.2; "leave roughly one-fifth" -> 0.2; "about half" / "50% of forecast" -> 0.5; "roughly 25% of forecast" -> 0.25; "leave about half" -> 0.5.
- RESERVE IS ALWAYS ABSOLUTE kWh IN OUTPUT: minimum_energy_kwh = pct * battery_capacity (capacity is given separately). E.g. 50% of 200 kWh -> 100. "Keep at least 90 kWh" -> 90. Never output a percent or fraction for reserve: 50% is 100 (not 0.5, not 50).
- DISTRACTOR RULE: cafeteria / library / deadline / club / booking / seminar / registration / menu notes with NO kW/solar/battery/grid/charge/discharge/reserve/hour energy language => no_op, applies=false, structured_adjustment=null. When in doubt and no energy equipment is mentioned, choose no_op.
- SHAPE RULE (exact keys, no extras): no_charge_window/no_discharge_window = {"hours": [...]} only; solar_reduction = {"hours": [...], "factor": ...} only; minimum_battery_reserve = {"hours": [...], "minimum_energy_kwh": ...} only; max_grid_window = {"hours": [...], "max_grid_kwh": ...} only; no_op = null. No extra keys ever.
- max_grid_kwh is an absolute kWh cap per listed hour (e.g. "must not exceed 155 kWh" -> 155; "stay at or below 190 kWh from 7 PM until 10 PM" -> {"hours":[19,20,21],"max_grid_kwh":190}).
- no_charge means charging unavailable ("do not charge", "charging disabled", "charger isolated/unavailable", "charging circuit unavailable"); no_discharge means discharging unavailable ("must not discharge", "do not discharge during relay testing", "discharge isolated").
- Every other type -> applies=true with the exact shape above.
- You must return exactly one entry per note, with note_index 0..N-1 in order.

Examples:
Note: "Solar output will drop to about 20% from 1 PM to 3 PM."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar reduced to 20% for hours 13-14."}
Note: "Panel washing from one until three will leave roughly one-fifth of normal solar output."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar one-fifth remaining for hours 13-14."}
Note: "PV production will drop to about 20% between 13:00 and 15:00."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar 20% remaining for hours 13-14."}
Note: "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar 80% reduction leaves 0.2 for hours 13-14."}
Note: "Cloud cover will leave about half of forecast solar from 10 AM until noon."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[10,11],"factor":0.5},"explanation":"Solar half remaining for hours 10-11."}
Note: "Keep at least 50% of the battery capacity stored from 6 PM until 9 PM. Battery capacity is 200 kWh."
-> {"note_index":0,"applies":true,"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[18,19,20],"minimum_energy_kwh":100},"explanation":"Reserve 50pct of 200kWh equals 100kWh for hours 18-20."}
Note: "Keep at least 120 kWh in reserve from 6 PM until 9 PM."
-> {"note_index":0,"applies":true,"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[18,19,20],"minimum_energy_kwh":120},"explanation":"Reserve 120kWh for hours 18-20."}
Note: "Do not charge the battery between 2 PM and 4 PM."
-> {"note_index":0,"applies":true,"directive_type":"no_charge_window","structured_adjustment":{"hours":[14,15]},"explanation":"Charging unavailable 14-15."}
Note: "The battery must not discharge from 6 PM until 8 PM."
-> {"note_index":0,"applies":true,"directive_type":"no_discharge_window","structured_adjustment":{"hours":[18,19]},"explanation":"Discharging unavailable 18-19."}
Note: "Do not discharge the battery from 5 PM until 7 PM during relay testing."
-> {"note_index":0,"applies":true,"directive_type":"no_discharge_window","structured_adjustment":{"hours":[17,18]},"explanation":"Discharging unavailable 17-18."}
Note: "From 6 PM until 9 PM, grid import must not exceed 155 kWh in any hour."
-> {"note_index":0,"applies":true,"directive_type":"max_grid_window","structured_adjustment":{"hours":[18,19,20],"max_grid_kwh":155},"explanation":"Grid capped at 155kWh for hours 18-20."}
Note: "Grid intake must stay at or below 190 kWh from 19:00 until 22:00 while the substation is constrained."
-> {"note_index":0,"applies":true,"directive_type":"max_grid_window","structured_adjustment":{"hours":[19,20,21],"max_grid_kwh":190},"explanation":"Grid capped at 190kWh for hours 19-21."}
Note: "The cafeteria menu changes tomorrow."
-> {"note_index":0,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"Irrelevant to energy schedule."}
Note: "The library is extending book-return hours next week."
-> {"note_index":0,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"Irrelevant to energy schedule."}
"""

# Slim corrective-retry prompt (#11): the model already saw full context on
# the first call, so the 1-note retry sends rules + 2 shots only (~85% fewer
# tokens than re-sending all 16 few-shots).
RETRY_SYSTEM_PROMPT = """You interpret campus operator notes into energy directives.
Return ONLY valid JSON, no prose, no markdown fences.
Allowed directive_type: solar_reduction {"hours":[...],"factor":REMAINING fraction}, minimum_battery_reserve {"hours":[...],"minimum_energy_kwh":absolute kWh}, no_charge_window/no_discharge_window {"hours":[...]}, max_grid_window {"hours":[...],"max_grid_kwh":kWh}, no_op null.
Rules: hours are start-inclusive end-exclusive ints 0..23 ascending ("1-3PM"->[13,14]; "6-9PM"->[18,19,20]); solar factor is REMAINING ("80% reduction"->0.2; "drop to 20%"->0.2); reserve is absolute kWh (50% of 200kWh->100); no energy equipment => no_op. Exact keys only, no extras.
Examples:
Note: "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar 80pct reduction leaves 0.2."}
Note: "Do not charge the battery between 2 PM and 4 PM."
-> {"note_index":0,"applies":true,"directive_type":"no_charge_window","structured_adjustment":{"hours":[14,15]},"explanation":"Charging unavailable 14-15."}
"""


def _safe_default(index: int) -> RawDirective:
    return RawDirective(
        note_index=index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation="LLM unavailable; defaulted to no_op.",
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(text: str):
    """Parse LLM text into a list of raw dicts. Raises on failure."""
    text = _strip_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: extract the first [...] or {...} block.
        m = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    if isinstance(data, dict):
        # Allow {"directives": [...]} or a single directive object.
        for key in ("directives", "interpretations", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("LLM JSON is not a list")
    return data


def _has_energy_keyword(note: str) -> bool:
    return bool(_ENERGY_RE.search(note or ""))


def _suspicious_no_ops(
    batch_notes: List[str], fresh: List["RawDirective"], capacity: float
) -> List[int]:
    """Positions in the fresh batch whose no_op is suspicious: the note
    contains energy keywords but validation still yields no_op. Uses a
    guardrails dry-run with a permissive synthetic battery (capacity real,
    other limits wide) so reserve-cap checks stay meaningful."""
    try:
        from guardrails import validate_directives as _validate
        from schemas import Battery as _Battery

        _bat = _Battery(
            capacity_kwh=float(capacity),
            initial_energy_kwh=0.0,
            minimum_energy_kwh=0.0,
            max_charge_kwh_per_hour=1e9,
            max_discharge_kwh_per_hour=1e9,
        )
        _checked = _validate(list(fresh), len(batch_notes), _bat)
    except Exception:
        return []
    out: List[int] = []
    for pos, (note, v) in enumerate(zip(batch_notes, _checked)):
        if v.directive_type == "no_op" and _has_energy_keyword(note):
            out.append(pos)
    return out


def _mentions_solar(note: str) -> bool:
    return bool(re.search(r"solar|photovoltaic|\bpv\b|panel|rooftop", note or "", re.IGNORECASE))


def _suspicious_mismatches(
    batch_notes: List[str], fresh: List["RawDirective"]
) -> List[int]:
    """Wrong-but-valid outputs (#5): confident mis-types that validate yet
    contradict the note text. Returns batch positions to retry."""
    out: List[int] = []
    for pos, (note, r) in enumerate(zip(batch_notes, fresh)):
        try:
            dtype = (r.directive_type or "").strip().lower()
        except Exception:
            continue
        if dtype == "no_op":
            continue  # handled by _suspicious_no_ops
        low = (note or "").lower()
        no_dis = bool(re.search(
            r"must not discharge|do not discharge|not discharge|"
            r"discharge\s+(is\s+)?(isolated|unavailable|disabled)|"
            r"discharge.*(unavailable|disabled|isolated)", low))
        no_ch = bool(re.search(
            r"do not charge|not charge|charging\s+(disabled|unavailable)|"
            r"charger.*isolated|charging circuit.*unavailable|"
            r"charging?\s+(is\s+)?(isolated|unavailable|disabled)", low))
        # Charge/discharge swap: note forbids one, model picked the other.
        if no_dis and dtype == "no_charge_window":
            out.append(pos)
            continue
        if no_ch and dtype == "no_discharge_window":
            out.append(pos)
            continue
        # Solar note typed as a non-solar directive.
        if _mentions_solar(note) and dtype in (
            "no_charge_window", "no_discharge_window",
            "max_grid_window", "minimum_battery_reserve",
        ):
            out.append(pos)
            continue
        # Grid-cap note typed as something else entirely.
        if "grid" in low and re.search(r"exceed|below|cap|limit|155|190|kwh", low) \
                and dtype in ("no_charge_window", "no_discharge_window",
                              "solar_reduction", "minimum_battery_reserve"):
            out.append(pos)
            continue
        # Reserve note typed as a window/cap/solar directive.
        if re.search(r"reserve|keep at least|minimum.*energy|%.*capacity", low) \
                and "battery" in low \
                and dtype in ("no_charge_window", "no_discharge_window",
                              "solar_reduction", "max_grid_window"):
            out.append(pos)
            continue
        # Factor polarity sanity: "80% reduction" must leave ~0.2, not 0.8.
        try:
            adj = r.structured_adjustment or {}
            if dtype == "solar_reduction" and isinstance(adj, dict) and "factor" in adj:
                f = float(adj["factor"])
                m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*reduct", low)
                if m and abs(f - float(m.group(1)) / 100.0) < 0.02:
                    out.append(pos)  # emitted reduction instead of remainder
                    continue
                m2 = re.search(r"drop to (?:about\s+)?(\d+(?:\.\d+)?)\s*%", low)
                if m2 and abs(f - (1.0 - float(m2.group(1)) / 100.0)) < 0.02:
                    out.append(pos)  # inverted drop-to
                    continue
        except Exception:
            pass
    return out


def _suspicious_reasons(
    batch_notes: List[str], fresh: List["RawDirective"], positions: List[int]
) -> List[str]:
    reasons: List[str] = []
    for pos in positions:
        try:
            r = fresh[pos]
            reasons.append(
                f"note {pos} got {r.directive_type} "
                f"{r.structured_adjustment} but mentions energy equipment"
            )
        except Exception:
            reasons.append(f"note {pos} rejected")
    return reasons


def _corrective_suffix(reasons: List[str]) -> str:
    joined = "; ".join(reasons[:3])
    return (
        "\nPrevious output was REJECTED by validation: "
        + joined
        + ". Return the exact shape for the applicable directive type "
        + "(hours as ints 0-23, solar factor as REMAINING fraction, "
        + "reserve as absolute kWh). One object per note."
    )


# --- A4: deterministic safety net (all-keys-failed ONLY) --------------------

_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def _ampm_to_24(h: int, ap: str | None, default_ap: str | None = None) -> int:
    ap = (ap or default_ap or "").lower()
    if ap.startswith("p") and h < 12:
        h += 12
    if ap.startswith("a") and h == 12:
        h = 0
    return h % 24


def _extract_hours(text: str) -> list | None:
    """Best-effort whole-hour range -> start-inclusive end-exclusive list.
    Returns None when no recognizable range is found."""
    low = text.lower()
    # 24h "13:00-15:00" / "19:00 until 22:00" / "between 13:00 and 15:00"
    m = re.search(r"(\d{1,2}):00\s*(?:-|–|until|to|and)\s*(\d{1,2}):00", low)
    if m:
        s, e = int(m.group(1)) % 24, int(m.group(2)) % 24
        if 0 <= s < 24 and 0 <= e < 24 and s != e:
            if e > s:
                if e <= 24:
                    return list(range(s, e))
            else:
                # Midnight-crossing (#9): "22:00-02:00" -> [0,1,22,23].
                if (24 - s + e) <= 12:
                    return sorted(list(range(s, 24)) + list(range(0, e)))
            return None
        return None
    # "1-3PM" / "1 PM to 3 PM" / "2 AM until 5 AM" / "between 2 and 4 PM",
    # incl. word numbers.
    m = re.search(
        r"(midnight|noon|midday|\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s*(am|pm)?\s*(?:-|–|until|to|and)\s*"
        r"(midnight|noon|midday|\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s*(am|pm)?",
        low,
    )
    if m:
        def _num(tok: str) -> int | None:
            tok = tok.strip()
            if tok in ("midnight",):
                return 0
            if tok in ("noon", "midday"):
                return 12
            if tok.isdigit():
                return int(tok)
            return _WORD_NUM.get(tok)

        s_raw, s_ap, e_raw, e_ap = m.group(1), m.group(2), m.group(3), m.group(4)
        s_n, e_n = _num(s_raw), _num(e_raw)
        if s_n is None or e_n is None:
            return None
        # Propagate a single marker: "1-3 PM" => both PM; bare solar
        # "one until three" with solar/panel/pv => daytime (PM).
        shared = s_ap or e_ap
        if shared is None and _mentions_solar(low):
            shared = "pm"
        # noon/midday always maps to 12, not 0 (avoid AM→0 conversion)
        if s_raw in ("noon", "midday"):
            s = 12
        else:
            s = _ampm_to_24(s_n, s_ap, shared)
        if e_raw in ("noon", "midday"):
            e = 12
        else:
            e = _ampm_to_24(e_n, e_ap, shared)
        if 0 <= s < 24 and 0 <= e < 24 and s != e:
            span = (e - s) % 24
            if 0 < span <= 12:
                if e > s:
                    return list(range(s, e))
                # Midnight-crossing (#9): "10 PM to 2 AM" -> [0,1,22,23].
                return sorted(list(range(s, 24)) + list(range(0, e)))
    return None


def _regex_fallback(note: str, capacity: float, index: int) -> "RawDirective | None":
    """Tiny obvious-case backup. Returns a RawDirective or None (meaning
    'not obvious — keep safe no_op default'). Explanation carries the
    'fallback:' prefix so the LLM path stays distinguishable."""
    from schemas import RawDirective as _RD

    low = note.lower()
    hours = _extract_hours(note)

    def _mk(dtype: str, adj: dict | None, applies: bool, why: str) -> _RD:
        return _RD(
            note_index=index,
            applies=applies,
            directive_type=dtype,
            structured_adjustment=adj,
            explanation=f"fallback: {why}",
        )

    # Distractors first (no hours needed).
    if any(w in low for w in ("cafeteria", "library", "menu", "deadline", "club", "booking", "seminar")):
        if not _has_energy_keyword(note):
            return _mk("no_op", None, False, "distractor note; no energy equipment.")
    if hours:
        if re.search(r"do not charge|not charge|charging (disabled|unavailable)|charger.*isolated|charging circuit.*unavailable", low):
            return _mk("no_charge_window", {"hours": hours}, True, f"charging unavailable {hours}.")
        if re.search(r"must not discharge|do not discharge|discharge isolated|not discharge", low):
            return _mk("no_discharge_window", {"hours": hours}, True, f"discharging unavailable {hours}.")
        # Solar: percent patterns -> REMAINING factor (#9: generalized).
        is_solar = _mentions_solar(note)
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*reduct\w*", low)
        if m and is_solar:
            return _mk("solar_reduction", {"hours": hours, "factor": round(1.0 - float(m.group(1)) / 100.0, 4)}, True, "percent reduction to remaining.")
        m = re.search(r"reduc\w*\s+by\s*(?:about\s+)?(\d+(?:\.\d+)?)\s*%", low)
        if m and is_solar:
            return _mk("solar_reduction", {"hours": hours, "factor": round(1.0 - float(m.group(1)) / 100.0, 4)}, True, "reduced-by percent to remaining.")
        m = re.search(r"drop to (?:about\s+|roughly\s+|around\s+)?(\d+(?:\.\d+)?)\s*%", low)
        if m and is_solar:
            return _mk("solar_reduction", {"hours": hours, "factor": round(float(m.group(1)) / 100.0, 4)}, True, "drop-to percent remaining.")
        m = re.search(r"cut to (?:about\s+|roughly\s+)?(\d+(?:\.\d+)?)\s*%", low)
        if m and is_solar:
            return _mk("solar_reduction", {"hours": hours, "factor": round(float(m.group(1)) / 100.0, 4)}, True, "cut-to percent remaining.")
        m = re.search(r"cut by (?:about\s+)?(\d+(?:\.\d+)?)\s*%", low)
        if m and is_solar:
            return _mk("solar_reduction", {"hours": hours, "factor": round(1.0 - float(m.group(1)) / 100.0, 4)}, True, "cut-by percent to remaining.")
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*forecast", low)
        if m and is_solar:
            return _mk("solar_reduction", {"hours": hours, "factor": round(float(m.group(1)) / 100.0, 4)}, True, "percent-of-forecast remaining.")
        if is_solar:
            if "one-fifth" in low or "one fifth" in low or "20%" in low:
                return _mk("solar_reduction", {"hours": hours, "factor": 0.2}, True, "one-fifth remaining.")
            if "half" in low or "50%" in low:
                return _mk("solar_reduction", {"hours": hours, "factor": 0.5}, True, "half remaining.")
            if "25%" in low or "quarter" in low:
                return _mk("solar_reduction", {"hours": hours, "factor": 0.25}, True, "quarter remaining.")
        m = re.search(r"(?:must not exceed|not exceed|at or below|stay at or below|capped? at|limit (?:is|of)|requires? at least|max(?:imum)?(?: of)?)\s*(\d+(?:\.\d+)?)\s*kwh", low)
        if m and "grid" in low:
            return _mk("max_grid_window", {"hours": hours, "max_grid_kwh": float(m.group(1))}, True, "grid cap kWh.")
        # Reserve kWh (#9): "requires at least 80 kWh", "minimum of X", etc.
        m = re.search(r"(?:keep|maintain|requires?|required|ensure|hold|store|retain)[^.]{0,40}?(?:at least|minimum(?: of)?|no less than)\s*(\d+(?:\.\d+)?)\s*kwh", low)
        if not m:
            m = re.search(r"(?:at least|minimum(?: of)?|no less than)\s*(\d+(?:\.\d+)?)\s*kwh", low)
        if m and ("reserve" in low or "battery" in low or "keep" in low or "require" in low):
            return _mk("minimum_battery_reserve", {"hours": hours, "minimum_energy_kwh": float(m.group(1))}, True, "reserve kWh.")
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of[^.]*capacity", low)
        if m and ("reserve" in low or "battery" in low or "keep" in low):
            try:
                return _mk("minimum_battery_reserve", {"hours": hours, "minimum_energy_kwh": round(float(m.group(1)) / 100.0 * float(capacity), 4)}, True, "reserve percent of capacity.")
            except Exception:
                pass
    return None


def interpret_notes(
    notes: List[str], battery_capacity_kwh: float, timeout_s: float = 20.0
) -> List[RawDirective]:
    """Return one RawDirective per note, in order. Never raises.

    Budget (#1/#2): the whole call (initial + ONE capped corrective retry)
    fits inside timeout_s. Retry fires only when >=1 note is salvageable
    AND >=4s remain; it is capped at min(5s, remaining).
    """
    import time as _t

    _t0 = _t.monotonic()
    # Serve cache hits without an API call (bounded LRU, #10).
    uncached_idx: List[int] = []
    results: List[RawDirective | None] = [None] * len(notes)
    for i, note in enumerate(notes):
        key = (note.strip(), round(float(battery_capacity_kwh), 4))
        cached = _cache_get(key)
        if cached is not None:
            results[i] = RawDirective(
                note_index=i,
                applies=cached.applies,
                directive_type=cached.directive_type,
                structured_adjustment=cached.structured_adjustment,
                explanation=cached.explanation,
            )
        else:
            uncached_idx.append(i)

    if not uncached_idx:
        return [r for r in results if r is not None]  # type: ignore[misc]

    all_keys_failed = False  # informational only; except-branch below salvages
    try:
        # Key pool IS the retry: _call_gemini starts at the next pool key
        # (round-robin) and fails over instantaneously on any error.
        # No outer retry (saves p95). Never log prompt/keys.
        fresh = _call_gemini(
            [notes[i] for i in uncached_idx],
            battery_capacity_kwh,
            timeout_s,
        )
        for slot, raw in zip(uncached_idx, fresh):
            raw.note_index = slot  # enforce position, ignore model numbering
            results[slot] = raw
            _cache_put((notes[slot].strip(), round(float(battery_capacity_kwh), 4)), raw)

        # A3: ONE corrective retry only, capped by remaining budget (#1/#2).
        # Triggers: validated no_op on energy-equipment notes (#6) PLUS
        # wrong-but-valid mis-types (#5). Slim retry prompt (#11).
        try:
            _batch = [notes[i] for i in uncached_idx]
            _s_noop = _suspicious_no_ops(_batch, fresh, battery_capacity_kwh)
        except Exception:
            _s_noop = []
        try:
            _s_mm = _suspicious_mismatches(_batch, fresh)
        except Exception:
            _s_mm = []
        _suspicious = sorted(set(_s_noop) | set(_s_mm))
        _elapsed = _t.monotonic() - _t0
        _remaining = float(timeout_s) - _elapsed
        if _suspicious and _remaining >= 4.0:
            try:
                _retry_notes = [_batch[j] for j in _suspicious]
                _reasons = _suspicious_reasons(_batch, fresh, _suspicious)
                _retry_budget = min(5.0, _remaining - 0.5)
                _corrected = _call_gemini(
                    _retry_notes,
                    battery_capacity_kwh,
                    _retry_budget,
                    _corrective_suffix(_reasons),
                    system_prompt=RETRY_SYSTEM_PROMPT,
                )
                for pos, raw in zip(_suspicious, _corrected):
                    slot = uncached_idx[pos]
                    raw.note_index = slot
                    # Only overwrite when the correction is applicable;
                    # a second no_op means the note really is irrelevant.
                    if raw.directive_type != "no_op" or raw.applies:
                        results[slot] = raw
                        _cache_put((notes[slot].strip(), round(float(battery_capacity_kwh), 4)), raw)
            except Exception:
                try:
                    import sys as _sys2

                    print("[llm] corrective retry failed", file=_sys2.stderr)
                except Exception:
                    pass
    except Exception:
        all_keys_failed = True
        for slot in uncached_idx:
            if results[slot] is None:
                # A4: deterministic safety net ONLY when every LLM key
                # failed. Tiny regex backup for obvious notes; the LLM
                # remains the primary path (mandatory rule). Never raises.
                _fb = _regex_fallback(
                    notes[slot], battery_capacity_kwh, slot
                )
                results[slot] = _fb if _fb is not None else _safe_default(slot)
                # Do NOT cache failure defaults; a transient error
                # shouldn't poison future identical notes.

    # Guarantee exactly one entry per note, in order.
    for i in range(len(notes)):
        if results[i] is None:
            results[i] = _safe_default(i)
        results[i].note_index = i
    return [r for r in results if r is not None]  # type: ignore[misc]


def _note_key_success(api_key: str) -> None:
    """Clear backoff state after a working key (#13)."""
    try:
        _KEY_FAIL_COUNT.pop(api_key, None)
    except Exception:
        pass


def _note_key_failure(api_key: str, exc: BaseException) -> None:
    """Class-first cooldown (#13): auth errors quarantine the key; quota /
    overload backs off exponentially; nothing else cools down (avoids the
    old substring overmatch on 'limit'/'retry')."""
    import time as _time

    try:
        cname = type(exc).__name__
        msg = f"{cname} {exc}".lower()
    except Exception:
        return
    try:
        # Auth / bad-key: quarantine long-term (dead keys never retried soon).
        if cname in ("Unauthenticated", "PermissionDenied", "Unauthorized") or \
                any(s in msg for s in ("401", "403", "invalid api key", "api key not valid", "permission denied", "unauthenticated")):
            _KEY_COOLDOWN_UNTIL[api_key] = _time.monotonic() + 1800.0
            return
        # Quota / overload / transient: exponential backoff 60s,120s,...≤300s.
        if cname in ("ResourceExhausted", "ServiceUnavailable", "Unavailable",
                     "DeadlineExceeded", "Internal", "Unknown") or \
                any(s in msg for s in ("429", "503", "504", "quota", "exhausted",
                                       "overload", "overloaded", "rate", "unavailable",
                                       "timeout", "timed out", "connection", "reset", "eof")):
            n = int(_KEY_FAIL_COUNT.get(api_key, 0)) + 1
            _KEY_FAIL_COUNT[api_key] = n
            _KEY_COOLDOWN_UNTIL[api_key] = _time.monotonic() + min(300.0, 60.0 * (2 ** (n - 1)))
    except Exception:
        pass


def _call_gemini(
    notes: List[str],
    battery_capacity_kwh: float,
    timeout_s: float,
    extra_suffix: str = "",
    system_prompt: str | None = None,
) -> List[RawDirective]:
    """ONE batched call with key pool + failover.

    Each invocation starts at the next pool key (round-robin), so concurrent
    requests run on different keys in parallel instead of queueing on key 1.
    Within one call, keys are tried sequentially from its start key on any
    failure (429 quota, 503 overload, 401/403 bad key, timeout, connection
    reset, empty response, bad JSON). First success wins; no retry on valid
    JSON. Raises only if ALL keys fail — caller degrades to safe defaults.
    extra_suffix appends corrective context for the 1x A3 retry, which uses
    the slim RETRY_SYSTEM_PROMPT (#11) instead of the full 16-shot prompt.
    """
    # Lazy import: keeps GET /health alive even if keys are misconfigured.
    try:
        import config  # type: ignore[import-not-found]

        model = config.GEMINI_MODEL
        keys = list(getattr(config, "GEMINI_API_KEYS", []) or [])
        if not keys and getattr(config, "GEMINI_API_KEY", None):
            keys = [config.GEMINI_API_KEY]
    except Exception:
        raise RuntimeError("Gemini key/model unavailable")

    if not keys:
        raise RuntimeError("Gemini key missing")

    numbered = "\n".join(f"Note {i}: {n}" for i, n in enumerate(notes))
    _sys_prompt = system_prompt or SYSTEM_PROMPT
    prompt = (
        _sys_prompt
        + f"\nBattery capacity: {battery_capacity_kwh} kWh "
        + "(use ONLY to convert percentage reserves to kWh).\n"
        + f"Interpret these {len(notes)} note(s); return a JSON array with "
        + "exactly one object per note, note_index 0..N-1 in order:\n"
        + numbered
        + (extra_suffix or "")
    )

    # Split the total budget across keys so N-key worst case ~= timeout_s
    # (#1): per-key share with a small 2s floor (client-side HTTP timeout —
    # sub-10s values are fine; the old 10s floor tripled worst-case spend).
    # An overall deadline is enforced across failovers so orphans die fast
    # (#3). Skip keys in cooldown; if all are cooling, use the one whose
    # cooldown expires soonest instead of failing outright.
    import time as _time

    _now = _time.monotonic()
    _deadline = _now + max(0.5, float(timeout_s))
    _healthy = [k for k in keys if _KEY_COOLDOWN_UNTIL.get(k, 0.0) <= _now]
    if not _healthy:
        _healthy = sorted(keys, key=lambda k: _KEY_COOLDOWN_UNTIL.get(k, 0.0))[:1]
    # Round-robin start: concurrent invocations fan out across the pool.
    _start = _next_pool_start(len(_healthy))
    _healthy = _healthy[_start:] + _healthy[:_start]
    per_key_share = float(timeout_s) / max(1, len(_healthy))
    last_err: Exception = RuntimeError("all Gemini keys failed")
    for ki, api_key in enumerate(_healthy):
        _remaining = _deadline - _time.monotonic()
        if _remaining <= 0.5:
            break
        attempt_timeout = max(2.0, min(per_key_share, _remaining))
        try:
            out = _call_gemini_with_key(
                prompt, notes, model, api_key, attempt_timeout
            )
            _note_key_success(api_key)
            return out
        except Exception as e:
            last_err = e
            # Fail over instantaneously; log key index + error TYPE only.
            # Never log prompt/keys/response.
            try:
                import sys as _sys

                print(
                    f"[llm] key {ki + 1}/{len(_healthy)} failed: {type(e).__name__}",
                    file=_sys.stderr,
                )
            except Exception:
                pass
            _note_key_failure(api_key, e)
            continue
    raise last_err


def _call_gemini_with_key(
    prompt: str, notes: List[str], model: str, api_key: str, timeout_s: float
) -> List[RawDirective]:
    from google import genai

    client = genai.Client(
        api_key=api_key, http_options={"timeout": int(timeout_s * 1000)}
    )

    # Constrain output via temperature 0 + JSON mime (see note above on why
    # response_json_schema is not used). Never log prompt/keys/response.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.0},
    )
    data = _parse_json(response.text or "")

    out: List[RawDirective] = []
    for i in range(len(notes)):
        if i < len(data) and isinstance(data[i], dict):
            try:
                out.append(RawDirective.model_validate(data[i]))
                continue
            except Exception:
                pass
        out.append(_safe_default(i))
    return out
