"""
maintenance_cost.py
===================
Business Cost & Downtime Estimation Engine for the IoT Predictive
Maintenance platform.

This module is the **financial intelligence layer** of the business pipeline.
It sits downstream of the Maintenance Priority Engine and converts AI-generated
failure predictions into dollar-denominated business impact figures that
operations managers can act on immediately.

Pipeline position::

    ML Inference
        |
        v
    MaintenancePriorityEngine  (src.configs.business.maintenance_priority)
        |
        v
    CostEstimationEngine       (this module)
        |
        v
    CostReport / Business Report Generator

What this engine produces
-------------------------
For every machine evaluated it calculates:

* **Repair cost range**           -- parts + labour, keyed to priority level
  and adjusted for machine age and operating hours.
* **Replacement cost**            -- total asset replacement value, scaled by
  machine type and age depreciation.
* **Downtime hours**              -- estimated outage duration, probability-
  weighted and failure-mode adjusted.
* **Production loss**             -- downtime * machine-type-specific hourly
  production cost.
* **Preventive maintenance cost** -- flat cost of proactive servicing.
* **Estimated savings**           -- failure cost minus preventive cost.
* **ROI**                         -- return on preventive maintenance
  investment, expressed as a percentage.
* **Business risk level**         -- qualitative risk classification
  (VERY HIGH / HIGH / MEDIUM / LOW).
* **Recommendations**             -- plain-English, financially framed action
  items.

All cost parameters are declared as module-level constants so they can be
overridden from a configuration file or environment variables without touching
business logic.

Usage::

    from src.configs.business.maintenance_cost import (
        CostEstimationEngine,
        CostInput,
        CostReport,
    )
    from src.configs.business.maintenance_priority import Priority

    engine = CostEstimationEngine()

    inp = CostInput(
        machine_id="CONV-014",
        machine_type="Conveyor Belt Motor",
        priority=Priority.CRITICAL,
        failure_probability=0.94,
        machine_age_years=6,
        operating_hours=14000,
        last_service_days=72,
        prediction_result=True,
    )

    report = engine.estimate(inp)
    engine.display_report(report)

    # One-liner convenience wrapper (returns JSON-safe dict)
    from src.configs.business.maintenance_cost import estimate_costs
    result = estimate_costs("CONV-014", "Conveyor Belt Motor", Priority.CRITICAL, ...)

Part of the Infotact Solutions Data Science & Machine Learning Internship project.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of Priority to avoid a hard circular dependency.
# The Priority enum is small and stable; we re-use it from the priority module.
# ---------------------------------------------------------------------------
try:
    from src.configs.business.maintenance_priority import Priority
except ImportError:  # Fallback for standalone execution / testing
    from maintenance_priority import Priority  # type: ignore[no-redef]


# ===========================================================================
# CONSTANTS
# All monetary and time figures are expressed in USD and decimal hours.
# Change these values to calibrate for a specific factory or industry sector.
# ===========================================================================

# ---------------------------------------------------------------------------
# Repair cost bounds  (USD)  keyed by Priority string value
# ---------------------------------------------------------------------------
REPAIR_COST_RANGE: Dict[str, Tuple[float, float]] = {
    "CRITICAL": (1_200.0, 2_500.0),
    "HIGH":     (700.0,   1_200.0),
    "MEDIUM":   (300.0,   700.0),
    "LOW":      (50.0,    300.0),
}

# ---------------------------------------------------------------------------
# Downtime bounds (hours) keyed by Priority string value
# ---------------------------------------------------------------------------
DOWNTIME_RANGE: Dict[str, Tuple[float, float]] = {
    "CRITICAL": (12.0, 24.0),
    "HIGH":     (6.0,  12.0),
    "MEDIUM":   (3.0,  6.0),
    "LOW":      (0.5,  2.0),
}

# ---------------------------------------------------------------------------
# Hourly production cost (USD / hour) by machine type.
# Used to convert downtime hours into a production loss dollar value.
# Adjust these to match the actual facility's throughput economics.
# ---------------------------------------------------------------------------
HOURLY_PRODUCTION_COST: Dict[str, float] = {
    "cnc machine":           450.0,
    "hydraulic pump":        320.0,
    "air compressor":        280.0,
    "conveyor belt motor":   240.0,
    "steam turbine":         520.0,
    "industrial generator":  490.0,
    "pump":                  300.0,
    "motor":                 260.0,
    "compressor":            280.0,
    "turbine":               500.0,
    "generator":             480.0,
    "press":                 380.0,
    "robot":                 410.0,
    "default":               300.0,   # fallback for unknown types
}

# ---------------------------------------------------------------------------
# Machine criticality multiplier -- scales production loss upward for
# machines that are critical path items in a production line.
# ---------------------------------------------------------------------------
CRITICALITY_MULTIPLIER: Dict[str, float] = {
    "CRITICAL": 1.50,
    "HIGH":     1.25,
    "MEDIUM":   1.00,
    "LOW":      0.80,
}

# ---------------------------------------------------------------------------
# Base replacement cost (USD) by machine type.
# Actual replacement cost is then depreciated by machine age.
# ---------------------------------------------------------------------------
BASE_REPLACEMENT_COST: Dict[str, float] = {
    "cnc machine":           85_000.0,
    "hydraulic pump":        18_000.0,
    "air compressor":        12_000.0,
    "conveyor belt motor":   9_500.0,
    "steam turbine":        120_000.0,
    "industrial generator":  75_000.0,
    "pump":                  15_000.0,
    "motor":                 8_000.0,
    "compressor":            11_000.0,
    "turbine":              100_000.0,
    "generator":             70_000.0,
    "press":                 40_000.0,
    "robot":                 95_000.0,
    "default":               20_000.0,
}

# ---------------------------------------------------------------------------
# Preventive maintenance base cost (USD) by priority level.
# This represents the typical cost of a scheduled, proactive service call.
# ---------------------------------------------------------------------------
PREVENTIVE_COST_BASE: Dict[str, float] = {
    "CRITICAL": 1_200.0,
    "HIGH":     850.0,
    "MEDIUM":   450.0,
    "LOW":      150.0,
}

# ---------------------------------------------------------------------------
# Age depreciation -- asset value remaining after N years
# (linear depreciation assumed; fully deprecated at ASSET_LIFE_YEARS)
# ---------------------------------------------------------------------------
ASSET_LIFE_YEARS: int = 15

# ---------------------------------------------------------------------------
# Ageing surcharge on repair cost per year beyond recommended service life
# ---------------------------------------------------------------------------
AGE_SURCHARGE_PER_YEAR: float = 0.03   # 3% additional repair cost per year

# ---------------------------------------------------------------------------
# Operating-hour wear factor -- repairs cost more on heavily used machines
# Applied when operating hours exceed the HIGH_HOURS_THRESHOLD
# ---------------------------------------------------------------------------
HIGH_HOURS_THRESHOLD: float = 10_000.0
HOURS_WEAR_FACTOR: float = 0.10        # 10% additional repair cost

# ---------------------------------------------------------------------------
# Service overdue surcharge -- extra cost when last service is overdue
# ---------------------------------------------------------------------------
SERVICE_OVERDUE_DAYS: int = 60
SERVICE_OVERDUE_SURCHARGE: float = 0.08  # 8% additional repair cost

# ---------------------------------------------------------------------------
# Probability weight: scales downtime toward the upper bound as the
# failure probability increases toward 1.0.
# Downtime = low + (high - low) * failure_probability
# ---------------------------------------------------------------------------
# (no separate constant needed -- probability is used directly as weight)


# ===========================================================================
# ENUMERATIONS
# ===========================================================================


class BusinessRisk(str, Enum):
    """Qualitative business risk classification keyed to cost exposure."""

    VERY_HIGH = "VERY HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Risk level lookup keyed by Priority string value
_RISK_BY_PRIORITY: Dict[str, BusinessRisk] = {
    "CRITICAL": BusinessRisk.VERY_HIGH,
    "HIGH":     BusinessRisk.HIGH,
    "MEDIUM":   BusinessRisk.MEDIUM,
    "LOW":      BusinessRisk.LOW,
}


# ===========================================================================
# DATA CLASSES
# ===========================================================================


@dataclass
class CostInput:
    """
    Input record for the Cost & Downtime Estimation Engine.

    Attributes
    ----------
    machine_id : str
        Unique machine identifier (e.g. ``"CONV-014"``).
    machine_type : str
        Human-readable machine category used to look up production costs and
        replacement values (e.g. ``"Conveyor Belt Motor"``).
    priority : Priority
        Maintenance priority level, typically produced by
        :class:`~src.configs.business.maintenance_priority.MaintenancePriorityEngine`.
    failure_probability : float
        Predicted failure probability in ``[0.0, 1.0]``.
    prediction_result : bool
        Binary failure flag emitted by the ML model.
    machine_age_years : float, optional
        Machine age in years.  Influences repair cost and replacement value.
        Defaults to ``0.0`` (new machine).
    operating_hours : float, optional
        Cumulative operating hours on the machine.  High-hour machines attract
        a wear surcharge on repair costs.  Defaults to ``0.0``.
    last_service_days : int, optional
        Days since the last preventive service.  Overdue machines attract an
        additional repair cost surcharge.  Defaults to ``0``.
    """

    machine_id: str
    machine_type: str
    priority: Priority
    failure_probability: float
    prediction_result: bool

    machine_age_years: float = 0.0
    operating_hours: float = 0.0
    last_service_days: int = 0


@dataclass
class CostReport:
    """
    Structured financial impact report produced by the Cost Estimation Engine.

    Attributes
    ----------
    machine_id : str
        Mirrors :attr:`CostInput.machine_id`.
    machine_type : str
        Mirrors :attr:`CostInput.machine_type`.
    priority : Priority
        Maintenance priority level.
    failure_probability : float
        Input failure probability (retained for reporting).
    business_risk : BusinessRisk
        Qualitative risk classification.
    repair_cost : float
        Estimated repair cost in USD.
    replacement_cost : float
        Estimated full asset replacement cost in USD.
    downtime_hours : float
        Estimated production downtime in hours.
    production_loss : float
        Estimated production revenue loss in USD.
    total_failure_cost : float
        ``repair_cost + production_loss`` -- total financial exposure if
        the machine is allowed to fail.
    preventive_cost : float
        Estimated cost of proactive preventive maintenance in USD.
    estimated_savings : float
        ``total_failure_cost - preventive_cost`` -- net financial benefit
        of performing preventive maintenance now.
    roi_pct : float
        Return on preventive maintenance investment as a percentage::

            roi = (estimated_savings / preventive_cost) * 100

    recommendations : list[str]
        Financially framed, actionable maintenance recommendations.
    cost_breakdown : dict
        Itemised cost components for display and downstream reporting.
    """

    machine_id: str
    machine_type: str
    priority: Priority
    failure_probability: float
    business_risk: BusinessRisk
    repair_cost: float
    replacement_cost: float
    downtime_hours: float
    production_loss: float
    total_failure_cost: float
    preventive_cost: float
    estimated_savings: float
    roi_pct: float
    recommendations: List[str]
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def roi_label(self) -> str:
        """Human-readable ROI string (e.g. ``"563%"``)."""
        return f"{self.roi_pct:.0f}%"

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the report to a plain, JSON-safe dictionary.

        Returns
        -------
        dict
            All fields serialised to primitive Python types.
        """
        return {
            "machine_id":          self.machine_id,
            "machine_type":        self.machine_type,
            "priority":            self.priority.value,
            "failure_probability": round(self.failure_probability, 4),
            "business_risk":       self.business_risk.value,
            "repair_cost":         round(self.repair_cost, 2),
            "replacement_cost":    round(self.replacement_cost, 2),
            "downtime_hours":      round(self.downtime_hours, 2),
            "production_loss":     round(self.production_loss, 2),
            "total_failure_cost":  round(self.total_failure_cost, 2),
            "preventive_cost":     round(self.preventive_cost, 2),
            "estimated_savings":   round(self.estimated_savings, 2),
            "roi":                 self.roi_label,
            "roi_pct":             round(self.roi_pct, 1),
            "recommendations":     self.recommendations,
            "cost_breakdown":      {
                k: round(v, 2) for k, v in self.cost_breakdown.items()
            },
        }


# ===========================================================================
# INTERNAL HELPER FUNCTIONS  -- pure, single-responsibility
# ===========================================================================


def _normalised_machine_type(machine_type: str) -> str:
    """
    Return a lower-cased, stripped machine type key for dictionary lookups.

    Parameters
    ----------
    machine_type : str
        Raw machine type string as supplied by the caller.

    Returns
    -------
    str
        Normalised key suitable for constant dictionary lookups.
    """
    return machine_type.strip().lower()


def _lookup_hourly_cost(machine_type: str) -> float:
    """
    Return the hourly production cost for *machine_type*.

    Falls back to the ``"default"`` entry when the type is not found.

    Parameters
    ----------
    machine_type : str
        Raw machine type string.

    Returns
    -------
    float
        Hourly production cost in USD.
    """
    key = _normalised_machine_type(machine_type)
    # Try exact match first, then check if any known keyword is a substring
    if key in HOURLY_PRODUCTION_COST:
        return HOURLY_PRODUCTION_COST[key]
    for known_key, cost in HOURLY_PRODUCTION_COST.items():
        if known_key in key or key in known_key:
            return cost
    return HOURLY_PRODUCTION_COST["default"]


def _lookup_replacement_cost(machine_type: str) -> float:
    """
    Return the base replacement asset value for *machine_type*.

    Falls back to the ``"default"`` entry when the type is not found.

    Parameters
    ----------
    machine_type : str
        Raw machine type string.

    Returns
    -------
    float
        Base replacement cost in USD (before age depreciation).
    """
    key = _normalised_machine_type(machine_type)
    if key in BASE_REPLACEMENT_COST:
        return BASE_REPLACEMENT_COST[key]
    for known_key, cost in BASE_REPLACEMENT_COST.items():
        if known_key in key or key in known_key:
            return cost
    return BASE_REPLACEMENT_COST["default"]


def _age_multiplier(machine_age_years: float) -> float:
    """
    Compute a repair cost age multiplier.

    Every year adds ``AGE_SURCHARGE_PER_YEAR`` to the base repair cost
    as older machines are harder and more expensive to service.

    Parameters
    ----------
    machine_age_years : float
        Machine age in years.

    Returns
    -------
    float
        Multiplicative factor (>= 1.0).
    """
    return 1.0 + max(0.0, machine_age_years) * AGE_SURCHARGE_PER_YEAR


def _hours_multiplier(operating_hours: float) -> float:
    """
    Apply a wear surcharge on machines with high cumulative operating hours.

    Parameters
    ----------
    operating_hours : float
        Cumulative hours the machine has been in service.

    Returns
    -------
    float
        Multiplicative factor (1.0 for normal hours, 1 + HOURS_WEAR_FACTOR
        for high-hour machines).
    """
    return (1.0 + HOURS_WEAR_FACTOR) if operating_hours > HIGH_HOURS_THRESHOLD else 1.0


def _service_overdue_multiplier(last_service_days: int) -> float:
    """
    Apply a surcharge when the machine is overdue for a scheduled service.

    Parameters
    ----------
    last_service_days : int
        Days elapsed since the last maintenance service.

    Returns
    -------
    float
        Multiplicative factor (1.0 when service is current).
    """
    return (
        (1.0 + SERVICE_OVERDUE_SURCHARGE)
        if last_service_days > SERVICE_OVERDUE_DAYS
        else 1.0
    )


def _estimate_repair_cost(inp: CostInput) -> Tuple[float, Dict[str, float]]:
    """
    Estimate the total repair cost for a machine failure.

    The base repair cost is the probability-weighted midpoint of the
    priority-specific range.  Three multiplicative adjustments are applied:

    * Age surcharge -- older machines are costlier to repair.
    * Hours wear surcharge -- heavily used machines need more work.
    * Service-overdue surcharge -- neglected machines accumulate deferred
      maintenance costs.

    Parameters
    ----------
    inp : CostInput
        Validated cost estimation input record.

    Returns
    -------
    tuple[float, dict]
        ``(total_repair_cost, itemised_breakdown)``
    """
    priority_key = inp.priority.value
    low, high = REPAIR_COST_RANGE[priority_key]

    # Probability-weighted interpolation: higher probability -> closer to upper bound
    base_cost = low + (high - low) * inp.failure_probability

    age_mult = _age_multiplier(inp.machine_age_years)
    hours_mult = _hours_multiplier(inp.operating_hours)
    service_mult = _service_overdue_multiplier(inp.last_service_days)

    age_surcharge = base_cost * (age_mult - 1.0)
    hours_surcharge = base_cost * (hours_mult - 1.0)
    service_surcharge = base_cost * (service_mult - 1.0)

    total = base_cost + age_surcharge + hours_surcharge + service_surcharge

    breakdown = {
        "base_repair_cost":        round(base_cost, 2),
        "age_surcharge":           round(age_surcharge, 2),
        "hours_wear_surcharge":    round(hours_surcharge, 2),
        "service_overdue_surcharge": round(service_surcharge, 2),
    }
    return total, breakdown


def _estimate_replacement_cost(inp: CostInput) -> float:
    """
    Estimate the current market replacement cost after age depreciation.

    Uses straight-line depreciation to ``0`` over ``ASSET_LIFE_YEARS`` years.
    A minimum residual value of 10% of original cost is retained.

    Parameters
    ----------
    inp : CostInput
        Validated cost estimation input record.

    Returns
    -------
    float
        Estimated replacement cost in USD.
    """
    base = _lookup_replacement_cost(inp.machine_type)
    depreciation_fraction = min(inp.machine_age_years / ASSET_LIFE_YEARS, 0.90)
    residual_fraction = 1.0 - depreciation_fraction
    return max(base * residual_fraction, base * 0.10)


def _estimate_downtime_hours(inp: CostInput) -> float:
    """
    Estimate the expected production downtime in hours.

    The estimate is the probability-weighted interpolation between the lower
    and upper downtime bounds for the given priority level.

    Parameters
    ----------
    inp : CostInput
        Validated cost estimation input record.

    Returns
    -------
    float
        Estimated downtime in decimal hours.
    """
    priority_key = inp.priority.value
    low, high = DOWNTIME_RANGE[priority_key]
    return low + (high - low) * inp.failure_probability


def _estimate_production_loss(
    downtime_hours: float,
    inp: CostInput,
) -> float:
    """
    Estimate the production revenue loss caused by the downtime.

    Production loss = downtime_hours * hourly_production_cost
                    * criticality_multiplier

    Parameters
    ----------
    downtime_hours : float
        Estimated downtime from :func:`_estimate_downtime_hours`.
    inp : CostInput
        Validated cost estimation input record.

    Returns
    -------
    float
        Estimated production loss in USD.
    """
    hourly_cost = _lookup_hourly_cost(inp.machine_type)
    criticality = CRITICALITY_MULTIPLIER[inp.priority.value]
    return downtime_hours * hourly_cost * criticality


def _estimate_preventive_cost(inp: CostInput) -> float:
    """
    Estimate the cost of performing proactive preventive maintenance now.

    The base preventive cost is adjusted upward if the machine is overdue for
    servicing (accumulated work is more involved).

    Parameters
    ----------
    inp : CostInput
        Validated cost estimation input record.

    Returns
    -------
    float
        Estimated preventive maintenance cost in USD.
    """
    base = PREVENTIVE_COST_BASE[inp.priority.value]
    # If service is overdue, preventive maintenance will be more comprehensive
    if inp.last_service_days > SERVICE_OVERDUE_DAYS:
        overdue_factor = 1.0 + (
            (inp.last_service_days - SERVICE_OVERDUE_DAYS) / 365.0
        ) * 0.20   # up to +20% extra per year overdue
        base *= overdue_factor
    return base


def _compute_roi(
    estimated_savings: float,
    preventive_cost: float,
) -> float:
    """
    Compute the return on investment for preventive maintenance.

    ROI (%) = (estimated_savings / preventive_cost) * 100

    Guards against division by zero (returns 0.0 when ``preventive_cost``
    is zero or negative).

    Parameters
    ----------
    estimated_savings : float
        Net financial benefit from preventing the failure.
    preventive_cost : float
        Cost of the preventive maintenance action.

    Returns
    -------
    float
        ROI as a percentage.
    """
    if preventive_cost <= 0.0:
        return 0.0
    return (estimated_savings / preventive_cost) * 100.0


def _build_recommendations(
    inp: CostInput,
    report_data: Dict[str, float],
) -> List[str]:
    """
    Generate financially framed, actionable maintenance recommendations.

    Recommendations are built in four layers:

    1. **Priority-specific immediate action** -- the single most important step.
    2. **ROI framing** -- whether preventive maintenance is economically justified.
    3. **Cost-driver-specific actions** -- age, operating hours, service overdue.
    4. **Machine-type-specific actions** -- generic maintenance tasks for the
       equipment category.

    Parameters
    ----------
    inp : CostInput
        Validated cost estimation input record.
    report_data : dict
        Pre-computed financial figures (keys: ``"repair_cost"``,
        ``"production_loss"``, ``"preventive_cost"``, ``"roi_pct"``).

    Returns
    -------
    list[str]
        Ordered, deduplicated list of recommendation strings.
    """
    recs: List[str] = []
    seen: set = set()

    def _add(rec: str) -> None:
        if rec not in seen:
            recs.append(rec)
            seen.add(rec)

    priority_key = inp.priority.value
    repair = report_data["repair_cost"]
    production = report_data["production_loss"]
    preventive = report_data["preventive_cost"]
    roi = report_data["roi_pct"]

    # --- 1. Priority-specific immediate action ----------------------------
    if priority_key == "CRITICAL":
        _add(
            "Immediate maintenance required -- halt production and schedule "
            "emergency repair within 24 hours to avoid unplanned failure"
        )
        _add(
            f"Emergency repair estimated at ${repair:,.0f}; "
            f"inaction risks ${production:,.0f} in production losses"
        )
    elif priority_key == "HIGH":
        _add(
            "Schedule maintenance within the next working shift -- "
            "failure risk is high and escalating"
        )
        _add(
            f"Estimated repair cost ${repair:,.0f}; "
            f"production loss exposure ${production:,.0f} if failure occurs"
        )
    elif priority_key == "MEDIUM":
        _add(
            "Create a maintenance work order for this week -- "
            "proactive action now avoids a more costly unplanned failure"
        )
        _add(
            f"Preventive maintenance (${preventive:,.0f}) is significantly "
            f"cheaper than unplanned repair (${repair:,.0f} + downtime losses)"
        )
    else:  # LOW
        _add(
            "Continue monitoring -- no urgent financial exposure at this time"
        )
        _add(
            "Include in the next planned preventive maintenance cycle to "
            "maintain low operating costs"
        )

    # --- 2. ROI framing ---------------------------------------------------
    if roi > 200.0:
        _add(
            f"Preventive maintenance ROI is {roi:.0f}% -- "
            f"investing ${preventive:,.0f} now saves an estimated "
            f"${report_data['estimated_savings']:,.0f}"
        )
    elif roi > 50.0:
        _add(
            f"Preventive maintenance delivers a {roi:.0f}% ROI -- "
            "financially justified"
        )

    # --- 3. Cost-driver-specific recommendations -------------------------
    if inp.machine_age_years > ASSET_LIFE_YEARS * 0.7:
        _add(
            f"Machine is {inp.machine_age_years:.0f} years old "
            f"({inp.machine_age_years / ASSET_LIFE_YEARS * 100:.0f}% of service life used) "
            "-- evaluate replacement versus continued repair investment"
        )

    if inp.operating_hours > HIGH_HOURS_THRESHOLD:
        _add(
            f"High operating hours ({inp.operating_hours:,.0f} hrs) -- "
            "schedule comprehensive overhaul; wear surcharge applied to repair estimate"
        )

    if inp.last_service_days > SERVICE_OVERDUE_DAYS:
        overdue_by = inp.last_service_days - SERVICE_OVERDUE_DAYS
        _add(
            f"Service is {overdue_by} days overdue -- "
            "deferred maintenance is increasing both repair cost and failure risk"
        )

    # --- 4. Machine-type-specific actions --------------------------------
    mt_key = _normalised_machine_type(inp.machine_type)

    if any(k in mt_key for k in ("pump", "hydraulic")):
        _add("Inspect hydraulic seals and fluid levels for leakage")
        _add("Check pump impeller and housing for cavitation damage")

    elif any(k in mt_key for k in ("compressor", "air")):
        _add("Inspect compressor valves, piston rings, and intercoolers")
        _add("Verify air filter condition and replace if overdue")

    elif any(k in mt_key for k in ("conveyor", "belt", "motor")):
        _add("Inspect belt tension, alignment, and roller bearings")
        _add("Check motor windings and lubricate drive components")

    elif any(k in mt_key for k in ("turbine",)):
        _add("Inspect turbine blades, seals, and bearing clearances")
        _add("Verify lube oil system pressure and filter condition")

    elif any(k in mt_key for k in ("generator",)):
        _add("Inspect alternator windings and excitation system")
        _add("Verify cooling system and fuel/air delivery components")

    elif any(k in mt_key for k in ("cnc",)):
        _add("Inspect spindle bearings and lubrication delivery system")
        _add("Verify axis drive servo motors and ball-screw condition")

    else:
        _add("Inspect mechanical bearings for wear, contamination, or scoring")
        _add("Inspect lubrication system and replenish consumables")

    return recs


def _validate_cost_input(inp: CostInput) -> None:
    """
    Validate *inp* field ranges and raise informative errors on bad data.

    Parameters
    ----------
    inp : CostInput
        Input record to validate.

    Raises
    ------
    TypeError
        If ``machine_id`` or ``machine_type`` is not a non-empty string.
    ValueError
        If any numeric field is outside a physically plausible range.
    """
    if not isinstance(inp.machine_id, str) or not inp.machine_id.strip():
        raise TypeError("machine_id must be a non-empty string.")

    if not isinstance(inp.machine_type, str) or not inp.machine_type.strip():
        raise TypeError("machine_type must be a non-empty string.")

    if not (0.0 <= inp.failure_probability <= 1.0):
        raise ValueError(
            f"failure_probability must be in [0.0, 1.0]; "
            f"got {inp.failure_probability}"
        )

    if inp.machine_age_years < 0:
        raise ValueError(
            f"machine_age_years cannot be negative; got {inp.machine_age_years}"
        )

    if inp.operating_hours < 0:
        raise ValueError(
            f"operating_hours cannot be negative; got {inp.operating_hours}"
        )

    if inp.last_service_days < 0:
        raise ValueError(
            f"last_service_days cannot be negative; got {inp.last_service_days}"
        )


# ===========================================================================
# MAIN ENGINE CLASS
# ===========================================================================


class CostEstimationEngine:
    """
    Converts ML failure predictions into financial impact estimates.

    The engine is **stateless** -- all parameters live in module-level
    constants, making instances lightweight and thread-safe.

    Methods
    -------
    estimate(inp)
        Evaluate a single :class:`CostInput` and return a :class:`CostReport`.

    estimate_batch(records)
        Evaluate a list of :class:`CostInput` records and return a list of
        :class:`CostReport` objects sorted by descending ``total_failure_cost``.

    display_report(report)
        Pretty-print a professional business report to stdout.

    display_batch(reports)
        Print a fleet financial summary followed by individual reports.

    Examples
    --------
    >>> engine = CostEstimationEngine()
    >>> report = engine.estimate(cost_input)
    >>> engine.display_report(report)
    """

    # ANSI colour codes
    _COLOURS: Dict[str, str] = {
        "CRITICAL":  "\033[91m",   # bright red
        "HIGH":      "\033[93m",   # bright yellow
        "MEDIUM":    "\033[94m",   # bright blue
        "LOW":       "\033[92m",   # bright green
        "VERY HIGH": "\033[91m",
        "green":     "\033[92m",
        "cyan":      "\033[96m",
    }
    _RESET: str = "\033[0m"
    _BOLD: str = "\033[1m"

    # ------------------------------------------------------------------
    def estimate(self, inp: CostInput) -> CostReport:
        """
        Perform a full cost and downtime estimation for one machine.

        Processing steps
        ----------------
        1. Validate inputs.
        2. Estimate repair cost (with age, hours, service multipliers).
        3. Estimate replacement cost (with age depreciation).
        4. Estimate downtime hours (probability-weighted).
        5. Estimate production loss (downtime * hourly cost * criticality).
        6. Estimate preventive maintenance cost.
        7. Compute savings and ROI.
        8. Classify business risk.
        9. Generate recommendations.
        10. Assemble and return :class:`CostReport`.

        Parameters
        ----------
        inp : CostInput
            Validated cost estimation input record.

        Returns
        -------
        CostReport
            Fully populated financial impact report.

        Raises
        ------
        TypeError
            If ``machine_id`` or ``machine_type`` is not a non-empty string.
        ValueError
            If any numeric field is outside a physically plausible range.
        """
        logger.info(
            "Estimating costs for machine: %s  [%s | priority=%s | prob=%.4f]",
            inp.machine_id,
            inp.machine_type,
            inp.priority.value,
            inp.failure_probability,
        )

        # Step 1: Validate
        _validate_cost_input(inp)

        # Step 2: Repair cost
        repair_cost, cost_breakdown = _estimate_repair_cost(inp)
        logger.debug("Repair cost for %s: $%.2f", inp.machine_id, repair_cost)

        # Step 3: Replacement cost
        replacement_cost = _estimate_replacement_cost(inp)
        logger.debug("Replacement cost for %s: $%.2f", inp.machine_id, replacement_cost)

        # Step 4: Downtime hours
        downtime_hours = _estimate_downtime_hours(inp)
        logger.debug("Downtime estimate for %s: %.2f hours", inp.machine_id, downtime_hours)

        # Step 5: Production loss
        production_loss = _estimate_production_loss(downtime_hours, inp)
        logger.debug("Production loss for %s: $%.2f", inp.machine_id, production_loss)

        # Step 6: Preventive cost
        preventive_cost = _estimate_preventive_cost(inp)
        logger.debug("Preventive cost for %s: $%.2f", inp.machine_id, preventive_cost)

        # Step 7: Savings and ROI
        total_failure_cost = repair_cost + production_loss
        estimated_savings = max(total_failure_cost - preventive_cost, 0.0)
        roi_pct = _compute_roi(estimated_savings, preventive_cost)
        logger.debug("ROI for %s: %.1f%%", inp.machine_id, roi_pct)

        # Step 8: Business risk
        business_risk = _RISK_BY_PRIORITY[inp.priority.value]

        # Step 9: Recommendations
        report_data = {
            "repair_cost":      repair_cost,
            "production_loss":  production_loss,
            "preventive_cost":  preventive_cost,
            "roi_pct":          roi_pct,
            "estimated_savings": estimated_savings,
        }
        recommendations = _build_recommendations(inp, report_data)

        # Step 10: Assemble report
        cost_breakdown["production_loss"] = round(production_loss, 2)
        cost_breakdown["total_failure_cost"] = round(total_failure_cost, 2)
        cost_breakdown["preventive_cost"] = round(preventive_cost, 2)
        cost_breakdown["estimated_savings"] = round(estimated_savings, 2)

        report = CostReport(
            machine_id=inp.machine_id,
            machine_type=inp.machine_type,
            priority=inp.priority,
            failure_probability=inp.failure_probability,
            business_risk=business_risk,
            repair_cost=repair_cost,
            replacement_cost=replacement_cost,
            downtime_hours=downtime_hours,
            production_loss=production_loss,
            total_failure_cost=total_failure_cost,
            preventive_cost=preventive_cost,
            estimated_savings=estimated_savings,
            roi_pct=roi_pct,
            recommendations=recommendations,
            cost_breakdown=cost_breakdown,
        )

        logger.info(
            "Cost report for %s -- Risk: %s | Total Failure Cost: $%.2f | "
            "Preventive Cost: $%.2f | ROI: %.0f%%",
            inp.machine_id,
            business_risk.value,
            total_failure_cost,
            preventive_cost,
            roi_pct,
        )
        return report

    # ------------------------------------------------------------------
    def estimate_batch(self, records: List[CostInput]) -> List[CostReport]:
        """
        Estimate costs for a list of machines and return reports sorted by
        descending total failure cost exposure.

        Parameters
        ----------
        records : list[CostInput]
            One :class:`CostInput` per machine.

        Returns
        -------
        list[CostReport]
            Reports sorted from highest to lowest ``total_failure_cost``.
        """
        if not records:
            logger.warning(
                "estimate_batch called with an empty list -- returning []."
            )
            return []

        logger.info("Estimating costs for a batch of %d machine(s).", len(records))
        reports = [self.estimate(r) for r in records]
        reports.sort(key=lambda r: r.total_failure_cost, reverse=True)
        return reports

    # ------------------------------------------------------------------
    def display_report(self, report: CostReport) -> None:
        """
        Print a professional, human-readable business cost report to stdout.

        Output format mirrors an enterprise maintenance management system
        report card with all key financial metrics and recommendations.

        Parameters
        ----------
        report : CostReport
            The cost report to display.
        """
        bold = self._BOLD
        reset = self._RESET
        priority_colour = self._COLOURS.get(report.priority.value, "")
        risk_colour = self._COLOURS.get(report.business_risk.value, "")
        cyan = self._COLOURS["cyan"]

        W = 60   # column width

        def _line(char: str = "=") -> str:
            return char * W

        def _row(label: str, value: str) -> None:
            print(f"  {label:<28} {value}")

        print()
        print(f"{bold}{_line('=')}{reset}")
        print(f"{bold}{'  Business Maintenance Cost Report':^{W}}{reset}")
        print(f"{bold}{_line('=')}{reset}")
        print()

        # --- Machine header -----------------------------------------------
        machine_label = f"{report.machine_id}  [{report.machine_type}]"
        _row("Machine:", f"{bold}{machine_label}{reset}")
        _row(
            "Priority:",
            f"{priority_colour}{bold}{report.priority.value}{reset}",
        )
        _row(
            "Failure Probability:",
            f"{bold}{report.failure_probability * 100:.1f}%{reset}",
        )
        _row(
            "Business Risk:",
            f"{risk_colour}{bold}{report.business_risk.value}{reset}",
        )

        print()
        print(f"  {_line('-')}")
        print(f"{bold}  Financial Impact Summary{reset}")
        print(f"  {_line('-')}")

        # --- Cost figures ------------------------------------------------
        _row(
            "Estimated Repair Cost:",
            f"{cyan}{bold}${report.repair_cost:>10,.0f}{reset}",
        )
        _row(
            "Estimated Downtime:",
            f"{bold}{report.downtime_hours:.1f} Hours{reset}",
        )
        _row(
            "Production Loss:",
            f"{cyan}{bold}${report.production_loss:>10,.0f}{reset}",
        )
        _row(
            "Total Failure Cost Exposure:",
            f"{priority_colour}{bold}${report.total_failure_cost:>10,.0f}{reset}",
        )
        _row(
            "Asset Replacement Value:",
            f"${report.replacement_cost:>10,.0f}",
        )

        print()
        print(f"  {_line('-')}")
        print(f"{bold}  Preventive Maintenance ROI Analysis{reset}")
        print(f"  {_line('-')}")

        _row(
            "Preventive Maintenance Cost:",
            f"{cyan}{bold}${report.preventive_cost:>10,.0f}{reset}",
        )
        _row(
            "Estimated Savings:",
            f"\033[92m{bold}${report.estimated_savings:>10,.0f}{reset}",
        )
        _row(
            "Return on Investment (ROI):",
            f"\033[92m{bold}{report.roi_label:>10}{reset}",
        )

        print()
        print(f"  {_line('-')}")
        print(f"{bold}  Cost Breakdown{reset}")
        print(f"  {_line('-')}")
        for component, amount in report.cost_breakdown.items():
            label = component.replace("_", " ").title()
            _row(f"  {label}:", f"${amount:>10,.2f}")

        print()
        print(f"  {_line('-')}")
        print(f"{bold}  Recommended Actions{reset}")
        print(f"  {_line('-')}")
        for idx, rec in enumerate(report.recommendations, start=1):
            # Word-wrap long recommendations at ~56 chars
            words = rec.split()
            line_buf: List[str] = []
            lines: List[str] = []
            for word in words:
                line_buf.append(word)
                if len(" ".join(line_buf)) > 54:
                    lines.append(" ".join(line_buf[:-1]))
                    line_buf = [word]
            if line_buf:
                lines.append(" ".join(line_buf))

            prefix = f"    {idx}. "
            indent = " " * len(prefix)
            for i, ln in enumerate(lines):
                print(f"{prefix if i == 0 else indent}{ln}")

        print()
        print(f"{bold}{_line('=')}{reset}")

    # ------------------------------------------------------------------
    def display_batch(self, reports: List[CostReport]) -> None:
        """
        Print a fleet-level financial summary followed by individual reports.

        Parameters
        ----------
        reports : list[CostReport]
            Pre-evaluated cost reports (typically sorted by
            :meth:`estimate_batch`).
        """
        if not reports:
            print("No cost reports to display.")
            return

        bold = self._BOLD
        reset = self._RESET
        W = 60

        # Fleet summary header
        print()
        print(f"{bold}{'=' * W}{reset}")
        print(f"{bold}{'  FLEET FINANCIAL EXPOSURE SUMMARY':^{W}}{reset}")
        print(f"{bold}{'=' * W}{reset}")
        print()

        total_failure_exposure = sum(r.total_failure_cost for r in reports)
        total_preventive_spend = sum(r.preventive_cost for r in reports)
        total_fleet_savings = sum(r.estimated_savings for r in reports)

        def _row(label: str, value: str) -> None:
            print(f"  {label:<32} {value}")

        _row("Machines Evaluated:", f"{bold}{len(reports)}{reset}")
        _row(
            "Total Failure Cost Exposure:",
            f"\033[91m{bold}${total_failure_exposure:>10,.0f}{reset}",
        )
        _row(
            "Total Preventive Spend:",
            f"\033[96m{bold}${total_preventive_spend:>10,.0f}{reset}",
        )
        _row(
            "Total Potential Savings:",
            f"\033[92m{bold}${total_fleet_savings:>10,.0f}{reset}",
        )

        fleet_roi = _compute_roi(total_fleet_savings, total_preventive_spend)
        _row("Fleet-Level ROI:", f"\033[92m{bold}{fleet_roi:.0f}%{reset}")

        print()
        print(f"{bold}  Machine Priority Distribution:{reset}")
        priority_counts: Dict[str, int] = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0
        }
        for rpt in reports:
            priority_counts[rpt.priority.value] += 1

        for prio, count in priority_counts.items():
            colour = self._COLOURS.get(prio, "")
            bar = "#" * count if count > 0 else "--"
            print(
                f"  {colour}{bold}{prio:<10}{reset}  {bar}  ({count})"
            )

        print()
        print(f"{bold}  Top Machines by Financial Exposure:{reset}")
        print(f"  {'Machine ID':<16} {'Priority':<10} {'Failure Cost':>14} {'Prev. Cost':>12} {'ROI':>8}")
        print(f"  {'-' * 62}")
        for rpt in reports[:10]:   # show top 10
            colour = self._COLOURS.get(rpt.priority.value, "")
            print(
                f"  {rpt.machine_id:<16} "
                f"{colour}{bold}{rpt.priority.value:<10}{reset} "
                f"${rpt.total_failure_cost:>12,.0f}  "
                f"${rpt.preventive_cost:>9,.0f}  "
                f"{rpt.roi_label:>7}"
            )

        print()
        print(f"{bold}  Detailed Reports (sorted by financial exposure):{reset}")
        for rpt in reports:
            self.display_report(rpt)


# ===========================================================================
# CONVENIENCE FUNCTION
# ===========================================================================


def estimate_costs(
    machine_id: str,
    machine_type: str,
    priority: Priority,
    failure_probability: float,
    prediction_result: bool,
    machine_age_years: float = 0.0,
    operating_hours: float = 0.0,
    last_service_days: int = 0,
) -> Dict[str, Any]:
    """
    Module-level convenience wrapper around :class:`CostEstimationEngine`.

    Accepts individual field values instead of a :class:`CostInput` object
    and returns a JSON-safe dictionary.  Mirrors the ``run_inference`` /
    ``evaluate_machine`` convenience patterns from other modules.

    Parameters
    ----------
    machine_id : str
        Unique machine identifier.
    machine_type : str
        Human-readable machine category.
    priority : Priority
        Maintenance priority level.
    failure_probability : float
        Predicted failure probability in ``[0.0, 1.0]``.
    prediction_result : bool
        Binary failure flag from the ML model.
    machine_age_years : float, optional
        Machine age in years.
    operating_hours : float, optional
        Cumulative operating hours.
    last_service_days : int, optional
        Days since last service.

    Returns
    -------
    dict
        JSON-safe dictionary with all cost and downtime estimates.
    """
    inp = CostInput(
        machine_id=machine_id,
        machine_type=machine_type,
        priority=priority,
        failure_probability=failure_probability,
        prediction_result=prediction_result,
        machine_age_years=machine_age_years,
        operating_hours=operating_hours,
        last_service_days=last_service_days,
    )
    engine = CostEstimationEngine()
    report = engine.estimate(inp)
    return report.to_dict()


# ===========================================================================
# DEMO EXECUTION
# Run as:  python -m src.configs.business.maintenance_cost
# ===========================================================================


def _run_demo() -> None:
    """
    Demonstrate the engine with a realistic fleet spanning all four priority
    levels, several machine types, and varying age / hours / service profiles.
    """
    import json
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = CostEstimationEngine()

    fleet: List[CostInput] = [
        # 1. CRITICAL -- old CNC with high hours and overdue service
        CostInput(
            machine_id="CNC-007",
            machine_type="CNC Machine",
            priority=Priority.CRITICAL,
            failure_probability=0.95,
            prediction_result=True,
            machine_age_years=9.5,
            operating_hours=18_500,
            last_service_days=78,
        ),
        # 2. CRITICAL -- hydraulic pump, escalated from HIGH by vibration
        CostInput(
            machine_id="PUMP-012",
            machine_type="Hydraulic Pump",
            priority=Priority.CRITICAL,
            failure_probability=0.78,
            prediction_result=True,
            machine_age_years=5.0,
            operating_hours=11_200,
            last_service_days=55,
        ),
        # 3. HIGH -- air compressor, overdue service and multiple anomalies
        CostInput(
            machine_id="COMP-034",
            machine_type="Air Compressor",
            priority=Priority.HIGH,
            failure_probability=0.62,
            prediction_result=True,
            machine_age_years=7.0,
            operating_hours=14_000,
            last_service_days=75,
        ),
        # 4. HIGH -- steam turbine, service overdue escalation
        CostInput(
            machine_id="TURB-019",
            machine_type="Steam Turbine",
            priority=Priority.HIGH,
            failure_probability=0.55,
            prediction_result=False,
            machine_age_years=4.5,
            operating_hours=9_800,
            last_service_days=68,
        ),
        # 5. MEDIUM -- conveyor belt motor, moderate risk
        CostInput(
            machine_id="CONV-014",
            machine_type="Conveyor Belt Motor",
            priority=Priority.MEDIUM,
            failure_probability=0.58,
            prediction_result=True,
            machine_age_years=3.0,
            operating_hours=7_200,
            last_service_days=42,
        ),
        # 6. LOW -- industrial generator, healthy machine
        CostInput(
            machine_id="GENR-002",
            machine_type="Industrial Generator",
            priority=Priority.LOW,
            failure_probability=0.21,
            prediction_result=False,
            machine_age_years=2.0,
            operating_hours=4_500,
            last_service_days=20,
        ),
    ]

    print()
    print("=" * 62)
    print("  IoT PREDICTIVE MAINTENANCE -- COST & DOWNTIME ENGINE DEMO")
    print("=" * 62)

    reports = engine.estimate_batch(fleet)
    engine.display_batch(reports)

    # Show JSON output for the top-exposure machine
    print()
    print("  [JSON Dict Output -- Highest Financial Exposure Machine]")
    print("-" * 62)
    top = reports[0].to_dict()
    print(json.dumps(top, indent=4))
    print("-" * 62)


if __name__ == "__main__":
    _run_demo()
