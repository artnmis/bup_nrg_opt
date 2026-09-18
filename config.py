"""Loads env vars. Nothing else in the codebase touches os.environ directly."""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY_2: str | None = os.environ.get("GEMINI_API_KEY_2")
GEMINI_API_KEY_3: str | None = os.environ.get("GEMINI_API_KEY_3")
GEMINI_API_KEY_4: str | None = os.environ.get("GEMINI_API_KEY_4")
GEMINI_API_KEY_5: str | None = os.environ.get("GEMINI_API_KEY_5")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Ordered pool with failover: key 1 -> 2 -> 3 -> 4 -> 5. Also honors a
# single comma-separated GEMINI_API_KEYS var if set (split, strip, drop
# empties). Deduplicated, order-preserving. May be empty when no key is
# configured.
_raw_extra: str = os.environ.get("GEMINI_API_KEYS", "") or ""
GEMINI_API_KEYS: list = []
for _candidate in (
    [GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4, GEMINI_API_KEY_5]
    + [p.strip() for p in _raw_extra.split(",")]
):
    if _candidate and _candidate.strip() and _candidate.strip() not in GEMINI_API_KEYS:
        GEMINI_API_KEYS.append(_candidate.strip())

# Do NOT raise at import: /health and module imports must stay alive even
# when keys are missing. Callers (llm_interpreter) treat missing keys as
# a controlled fallback to no_op directives.
