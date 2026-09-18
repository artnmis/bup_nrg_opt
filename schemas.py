"""Pydantic models — single source of truth for every field name and enum."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


class HourEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class Battery(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def _clamp_to_capacity(self) -> "Battery":
        """Cross-field guard (#12): initial/minimum above capacity would make
        every LP solve infeasible (wasting the 2x3s CBC budget). Clamp into
        [0, capacity] so the optimizer always gets a feasible box; the
        directive reserves still apply on top."""
        try:
            cap = float(self.capacity_kwh)
        except Exception:
            return self
        try:
            if float(self.initial_energy_kwh) > cap:
                self.initial_energy_kwh = cap
            if float(self.minimum_energy_kwh) > cap:
                self.minimum_energy_kwh = cap
        except Exception:
            pass
        return self


class OptimizeRequest(BaseModel):
    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def notes_non_empty(cls, v: List[str]) -> List[str]:
        for note in v:
            if not note or not note.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @model_validator(mode="after")
    def hours_cover_0_to_23(self) -> "OptimizeRequest":
        seen = sorted(h.hour for h in self.hours)
        if seen != list(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0..23")
        return self


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict]
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class RawDirective(BaseModel):
    """Internal: raw LLM guess per note, before guardrails.

    Never returned to the client directly — guardrails.py coerces this
    into a valid DirectiveInterpretation.
    """

    model_config = {"extra": "ignore"}

    note_index: int = 0
    applies: bool = False
    directive_type: str = "no_op"
    structured_adjustment: Optional[dict] = None
    explanation: str = ""
