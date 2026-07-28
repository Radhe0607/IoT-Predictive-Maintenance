"""
maintenance_priority.py
=======================
Maintenance Priority Engine for the IoT Predictive Maintenance platform.

This module is the **business-logic layer** that sits on top of the ML
inference pipeline.  Raw failure probabilities alone are not actionable for
a maintenance engineer; this engine enriches each prediction with:

  * A structured priority level  (CRITICAL / HIGH / MEDIUM / LOW)
  * A concrete maintenance window (e.g. "Within 24 Hours")
  * Business-oriented recommended actions  (e.g. "Replace worn bearing")
  * Plain-English reasons explaining *why* that priority was assigned

Architecture
------------
The engine is intentionally stateless and side-effect-free so that it can
be embedded in any context: REST APIs, batch-scoring pipelines, notebooks,
or CLI scripts.

Priority Assignment Logic
-------------------------
1. **Base priority** is derived from the failure probability::

       >= 0.90  ->  CRITICAL
       >= 0.75  ->  HIGH
       >= 0.50  ->  MEDIUM
        < 0.50  ->  LOW

2. **Risk escalation** bumps the priority one level up when any of the
   following conditions hold:

       * Temperature exceeds the configured threshold
       * Vibration exceeds the configured threshold ("very high")
       * The machine has not been serviced for > 60 days
       * Two or more sensor readings are simultaneously abnormal
       * The ML model predicted failure=True with high confidence (>= 0.80)

   The priority is capped at CRITICAL -- it can never exceed that level.

3. **Recommended actions** and **reasons** are assembled from the same
   contextual signals and formatted as plain, operator-friendly English.

Usage::

    from src.configs.business.maintenance_priority import (
        MaintenancePriorityEngine,
        MachineData,
    )

    engine = MaintenancePriorityEngine()

    data = MachineData(
        machine_id="CNC-007",
        failure_probability=0.92,
        temperature=88.5,
        pressure=5.1,
        vibration=9.2,
        last_service_days=70,
        prediction_result=True,
        machine_type="CNC Machine",
    )

    decision = engine.evaluate(data)
    engine.display(decision)

Part of the Infotact Solutions Data Science & Machine Learning Internship project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# CONSTANTS  -- all hard-coded thresholds live here, not scattered in logic
# ===========================================================================

# --- Probability thresholds ------------------------------------------------
PROB_CRITICAL: float = 0.90
PROB_HIGH: float = 0.75
PROB_MEDIUM: float = 0.50

# --- Sensor alert thresholds -----------------------------------------------
TEMP_THRESHOLD: float = 80.0       # degrees C  -- above this triggers escalation
VIBRATION_THRESHOLD: float = 8.0   # mm/s       -- above this triggers escalation
PRESSURE_HIGH: float = 10.0        # bar        -- abnormal upper bound
PRESSURE_LOW: float = 1.0          # bar        -- abnormal lower bound
HUMIDITY_HIGH: float = 85.0        # %RH        -- abnormal upper bound

# --- Service interval threshold --------------------------------------------
SERVICE_OVERDUE_DAYS: int = 60     # days since last service -- triggers escalation

# --- High-confidence failure threshold (used for escalation) ---------------
HIGH_CONFIDENCE_FAILURE_PROB: float = 0.80

# --- Minimum abnormal sensor count to trigger the "multiple anomalies" flag -
MULTI_ANOMALY_COUNT: int = 2


# ===========================================================================
# ENUMERATIONS
# ===========================================================================


class Priority(str, Enum):
    """Maintenance urgency levels, ordered from lowest (0) to highest (3)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def __lt__(self, other: "Priority") -> bool:  # noqa: D105
        return _PRIORITY_ORDER[self] < _PRIORITY_ORDER[other]

    def __le__(self, other: "Priority") -> bool:  # noqa: D105
        return _PRIORITY_ORDER[self] <= _PRIORITY_ORDER[other]

    def __gt__(self, other: "Priority") -> bool:  # noqa: D105
        return _PRIORITY_ORDER[self] > _PRIORITY_ORDER[other]

    def __ge__(self, other: "Priority") -> bool:  # noqa: D105
        return _PRIORITY_ORDER[self] >= _PRIORITY_ORDER[other]


# Ordinal mapping -- defined outside the class body to avoid Enum metaclass
# conflicts when using comparison operators.
_PRIORITY_ORDER: Dict[Priority, int] = {
    Priority.LOW: 0,
    Priority.MEDIUM: 1,
    Priority.HIGH: 2,
    Priority.CRITICAL: 3,
}

# Ordered ladder used by the escalation helper (index = ordinal)
_PRIORITY_LADDER: List[Priority] = [
    Priority.LOW,
    Priority.MEDIUM,
    Priority.HIGH,
    Priority.CRITICAL,
]

# Maintenance windows keyed by priority
_MAINTENANCE_WINDOWS: Dict[Priority, str] = {
    Priority.CRITICAL: "Within 24 Hours",
    Priority.HIGH: "Within 3 Days",
    Priority.MEDIUM: "Within 7 Days",
    Priority.LOW: "Monitor During Next Maintenance Cycle",
}


# ===========================================================================
# DATA CLASSES
# ===========================================================================


@dataclass
class MachineData:
    """
    Input record for the Maintenance Priority Engine.

    Attributes
    ----------
    machine_id : str
        Unique identifier for the machine (e.g. ``"PUMP-003"``).
    failure_probability : float
        Predicted failure probability in the range ``[0.0, 1.0]``.
    temperature : float
        Current temperature reading in degrees Celsius.
    pressure : float
        Current pressure reading in bar.
    vibration : float
        Current vibration level in mm/s.
    prediction_result : bool
        Binary failure prediction emitted by the ML model
        (``True`` = failure predicted).
    humidity : float, optional
        Relative humidity in percent (0-100).  ``None`` if unavailable.
    machine_type : str, optional
        Human-readable machine category (e.g. ``"Hydraulic Press"``).
        Used only for display and action wording.
    last_service_days : int, optional
        Number of days elapsed since the machine was last serviced.
        ``None`` if the information is unavailable.
    """

    machine_id: str
    failure_probability: float
    temperature: float
    pressure: float
    vibration: float
    prediction_result: bool

    humidity: Optional[float] = None
    machine_type: Optional[str] = None
    last_service_days: Optional[int] = None


@dataclass
class MaintenanceDecision:
    """
    Structured output produced by the Maintenance Priority Engine.

    Attributes
    ----------
    machine_id : str
        Mirrors :attr:`MachineData.machine_id`.
    priority : Priority
        Assigned maintenance priority level.
    maintenance_window : str
        Human-readable time window in which action must be taken.
    recommended_actions : list[str]
        Ordered list of concrete maintenance actions.
    reason : list[str]
        Plain-English explanations for the assigned priority.
    failure_probability : float
        The input probability, retained for downstream use and reporting.
    machine_type : str or None
        Mirrors :attr:`MachineData.machine_type`.

    Examples
    --------
    >>> decision.to_dict()
    {
        "machine_id": "CNC-007",
        "priority": "CRITICAL",
        "maintenance_window": "Within 24 Hours",
        "recommended_actions": ["...", "..."],
        "reason": ["...", "..."],
        "failure_probability": 0.95,
        "machine_type": "CNC Machine"
    }
    """

    machine_id: str
    priority: Priority
    maintenance_window: str
    recommended_actions: List[str]
    reason: List[str]
    failure_probability: float
    machine_type: Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the decision to a plain dictionary (JSON-safe)."""
        return {
            "machine_id": self.machine_id,
            "priority": self.priority.value,
            "maintenance_window": self.maintenance_window,
            "recommended_actions": self.recommended_actions,
            "reason": self.reason,
            "failure_probability": round(self.failure_probability, 4),
            "machine_type": self.machine_type,
        }


# ===========================================================================
# INTERNAL HELPERS  -- pure functions; each does exactly one thing
# ===========================================================================


def _base_priority(probability: float) -> Priority:
    """
    Derive the **base** priority level solely from failure probability.

    Parameters
    ----------
    probability : float
        Failure probability in ``[0.0, 1.0]``.

    Returns
    -------
    Priority
        One of CRITICAL / HIGH / MEDIUM / LOW.
    """
    if probability >= PROB_CRITICAL:
        return Priority.CRITICAL
    if probability >= PROB_HIGH:
        return Priority.HIGH
    if probability >= PROB_MEDIUM:
        return Priority.MEDIUM
    return Priority.LOW


def _escalate(priority: Priority) -> Priority:
    """
    Bump *priority* up by exactly one level, capped at CRITICAL.

    Parameters
    ----------
    priority : Priority
        Current priority level.

    Returns
    -------
    Priority
        Escalated priority -- never exceeds CRITICAL.
    """
    current_index = _PRIORITY_LADDER.index(priority)
    escalated_index = min(current_index + 1, len(_PRIORITY_LADDER) - 1)
    return _PRIORITY_LADDER[escalated_index]


def _count_abnormal_sensors(data: MachineData) -> int:
    """
    Count how many sensor readings are simultaneously outside normal bounds.

    Sensors evaluated
    -----------------
    * Temperature  > TEMP_THRESHOLD
    * Vibration    > VIBRATION_THRESHOLD
    * Pressure     outside ``[PRESSURE_LOW, PRESSURE_HIGH]``
    * Humidity     > HUMIDITY_HIGH  (only when the value is provided)

    Parameters
    ----------
    data : MachineData
        Input sensor record.

    Returns
    -------
    int
        Number of sensors currently reading outside their normal range.
    """
    count: int = 0
    if data.temperature > TEMP_THRESHOLD:
        count += 1
    if data.vibration > VIBRATION_THRESHOLD:
        count += 1
    if not (PRESSURE_LOW <= data.pressure <= PRESSURE_HIGH):
        count += 1
    if data.humidity is not None and data.humidity > HUMIDITY_HIGH:
        count += 1
    return count


def _build_escalation_flags(data: MachineData) -> List[str]:
    """
    Identify which escalation conditions are active for *data*.

    Returns a list of short string tags, each representing one active
    condition.  An empty list means no escalation is warranted.

    Tag vocabulary (internal use only)
    ------------------------------------
    ``"high_temp"``
        Temperature above threshold.
    ``"high_vibration"``
        Vibration above threshold.
    ``"service_overdue"``
        Machine overdue for servicing.
    ``"multi_anomaly"``
        Two or more sensors abnormal simultaneously.
    ``"high_conf_failure"``
        ML model predicted failure with high confidence.

    Parameters
    ----------
    data : MachineData
        Input sensor record.

    Returns
    -------
    list[str]
        Active escalation flag tags.
    """
    flags: List[str] = []

    if data.temperature > TEMP_THRESHOLD:
        flags.append("high_temp")

    if data.vibration > VIBRATION_THRESHOLD:
        flags.append("high_vibration")

    if (
        data.last_service_days is not None
        and data.last_service_days > SERVICE_OVERDUE_DAYS
    ):
        flags.append("service_overdue")

    if _count_abnormal_sensors(data) >= MULTI_ANOMALY_COUNT:
        flags.append("multi_anomaly")

    if (
        data.prediction_result
        and data.failure_probability >= HIGH_CONFIDENCE_FAILURE_PROB
    ):
        flags.append("high_conf_failure")

    return flags


def _build_reasons(
    data: MachineData,
    base_priority: Priority,
    escalation_flags: List[str],
    final_priority: Priority,
) -> List[str]:
    """
    Compose a list of plain-English sentences explaining the priority decision.

    Parameters
    ----------
    data : MachineData
        Input sensor record.
    base_priority : Priority
        Priority before escalation.
    escalation_flags : list[str]
        Active escalation condition tags from :func:`_build_escalation_flags`.
    final_priority : Priority
        Priority after any escalation has been applied.

    Returns
    -------
    list[str]
        Non-empty list of human-readable reason strings.
    """
    reasons: List[str] = []
    flag_set: Set[str] = set(escalation_flags)

    # --- Probability-based reason ------------------------------------------
    pct = round(data.failure_probability * 100, 1)
    reasons.append(
        f"Failure probability is {pct}% "
        f"(base classification: {base_priority.value})"
    )

    # --- Sensor-specific reasons -------------------------------------------
    if "high_temp" in flag_set:
        reasons.append(
            f"Temperature {data.temperature:.1f} C exceeds the "
            f"{TEMP_THRESHOLD:.0f} C safety threshold"
        )

    if "high_vibration" in flag_set:
        reasons.append(
            f"Vibration {data.vibration:.2f} mm/s exceeds the "
            f"{VIBRATION_THRESHOLD:.0f} mm/s threshold -- "
            "abnormal mechanical stress detected"
        )

    if "service_overdue" in flag_set:
        overdue_by = data.last_service_days - SERVICE_OVERDUE_DAYS  # type: ignore[operator]
        reasons.append(
            f"Machine has not been serviced for {data.last_service_days} days "
            f"(overdue by {overdue_by} days)"
        )

    if "multi_anomaly" in flag_set:
        anomaly_count = _count_abnormal_sensors(data)
        reasons.append(
            f"{anomaly_count} sensor(s) reading simultaneously outside normal "
            "bounds -- combined risk elevation applied"
        )

    if "high_conf_failure" in flag_set:
        reasons.append(
            f"ML model predicted FAILURE with {data.failure_probability * 100:.1f}% "
            "confidence -- high-confidence positive prediction"
        )

    # --- Escalation summary notice ----------------------------------------
    if flag_set and final_priority != base_priority:
        reasons.append(
            f"Priority escalated from {base_priority.value} to "
            f"{final_priority.value} due to risk factors above"
        )

    return reasons


def _build_actions(
    data: MachineData,
    final_priority: Priority,
    escalation_flags: List[str],
) -> List[str]:
    """
    Generate a prioritised list of concrete maintenance recommendations.

    Actions are assembled in three layers:

    1. **Priority-level universal actions** -- always present, ordered by urgency.
    2. **Sensor-specific actions** -- conditionally added based on active flags.
    3. **Anomaly-specific actions** -- pressure / humidity checks added when
       the corresponding readings are out of range.

    Parameters
    ----------
    data : MachineData
        Input sensor record.
    final_priority : Priority
        The resolved priority level after any escalation.
    escalation_flags : list[str]
        Active escalation condition tags.

    Returns
    -------
    list[str]
        Deduplicated, ordered list of action strings.
    """
    actions: List[str] = []
    seen: Set[str] = set()
    flag_set: Set[str] = set(escalation_flags)

    def _add(action: str) -> None:
        """Append *action* only if it has not already been added (dedup guard)."""
        if action not in seen:
            actions.append(action)
            seen.add(action)

    # --- 1. Priority-level universal actions --------------------------------
    if final_priority == Priority.CRITICAL:
        _add("Immediately halt machine operation and isolate the unit")
        _add("Notify maintenance supervisor and safety team without delay")
        _add("Conduct comprehensive root-cause inspection before restart")

    elif final_priority == Priority.HIGH:
        _add("Schedule immediate inspection within the next working shift")
        _add("Notify the maintenance team and escalate work order priority")
        _add("Restrict operating load until inspection is completed")

    elif final_priority == Priority.MEDIUM:
        _add("Create a scheduled maintenance work order for this week")
        _add(
            "Monitor machine performance with increased sensor "
            "sampling frequency"
        )

    else:  # LOW
        _add("Record observation in maintenance log")
        _add("Include in next planned preventive maintenance cycle")

    # --- 2. Sensor-specific actions -----------------------------------------
    if "high_temp" in flag_set:
        _add(
            "Inspect and clean cooling system "
            "(fans, heat exchangers, coolant levels)"
        )
        _add(
            "Inspect lubrication system -- "
            "inadequate lubrication causes overheating"
        )
        _add(
            "Verify that thermal relief valves are functioning correctly"
        )

    if "high_vibration" in flag_set:
        _add(
            "Inspect motor bearings for wear, scoring, or contamination"
        )
        _add(
            "Check shaft alignment and balance -- "
            "misalignment causes excess vibration"
        )
        _add(
            "Inspect coupling, belt, and gear components for damage"
        )
        _add(
            "Monitor vibration trend -- "
            "consider replacing worn bearing if condition persists"
        )

    if "service_overdue" in flag_set:
        _add(
            f"Perform full preventive maintenance -- unit is "
            f"{data.last_service_days} days since last service"
        )
        _add(
            "Replenish or replace lubricants, filters, and "
            "consumable components"
        )

    if "multi_anomaly" in flag_set:
        _add(
            "Perform multi-point sensor calibration to rule out "
            "false readings"
        )
        _add(
            "Conduct comprehensive system-wide inspection across "
            "all subsystems"
        )

    # --- 3. Point-anomaly checks (pressure / humidity) ----------------------
    if not (PRESSURE_LOW <= data.pressure <= PRESSURE_HIGH):
        if data.pressure > PRESSURE_HIGH:
            _add(
                f"Investigate high pressure ({data.pressure:.2f} bar) -- "
                "check relief valve and downstream blockages"
            )
        else:
            _add(
                f"Investigate low pressure ({data.pressure:.2f} bar) -- "
                "check for leaks, pump wear, or supply issues"
            )

    if data.humidity is not None and data.humidity > HUMIDITY_HIGH:
        _add(
            f"Humidity at {data.humidity:.1f}% -- inspect for condensation, "
            "seal integrity, and corrosion risk"
        )

    # --- Fallback (LOW with no flags and nothing triggered) -----------------
    if not actions:
        _add("Continue routine monitoring -- no immediate action required")

    return actions


def _validate_input(data: MachineData) -> None:
    """
    Validate *data* field ranges and raise informative errors on bad input.

    Parameters
    ----------
    data : MachineData
        Input sensor record to validate.

    Raises
    ------
    TypeError
        If ``machine_id`` is not a non-empty string.
    ValueError
        If any numeric field is outside a physically plausible range.
    """
    if not isinstance(data.machine_id, str) or not data.machine_id.strip():
        raise TypeError("machine_id must be a non-empty string.")

    if not (0.0 <= data.failure_probability <= 1.0):
        raise ValueError(
            f"failure_probability must be in [0.0, 1.0]; "
            f"got {data.failure_probability}"
        )

    if data.temperature < -50 or data.temperature > 1_000:
        raise ValueError(
            f"temperature out of plausible range (-50 to 1000 C); "
            f"got {data.temperature}"
        )

    if data.vibration < 0:
        raise ValueError(
            f"vibration cannot be negative; got {data.vibration}"
        )

    if data.pressure < 0:
        raise ValueError(
            f"pressure cannot be negative; got {data.pressure}"
        )

    if data.humidity is not None and not (0.0 <= data.humidity <= 100.0):
        raise ValueError(
            f"humidity must be in [0, 100] %; got {data.humidity}"
        )

    if data.last_service_days is not None and data.last_service_days < 0:
        raise ValueError(
            f"last_service_days cannot be negative; "
            f"got {data.last_service_days}"
        )


# ===========================================================================
# MAIN ENGINE CLASS
# ===========================================================================


class MaintenancePriorityEngine:
    """
    Converts ML failure predictions into structured maintenance decisions.

    The engine is **stateless** -- all configuration lives in module-level
    constants, making instances lightweight and thread-safe.

    Methods
    -------
    evaluate(data)
        Core evaluation method.  Accepts a :class:`MachineData` record and
        returns a :class:`MaintenanceDecision`.

    evaluate_batch(records)
        Evaluate a list of :class:`MachineData` records and return a list of
        :class:`MaintenanceDecision` objects, sorted by descending priority.

    display(decision)
        Pretty-print a single :class:`MaintenanceDecision` to stdout with
        colour-coded priority levels.

    display_batch(decisions)
        Pretty-print a full batch of decisions, preceded by a fleet summary
        table.

    Examples
    --------
    >>> engine = MaintenancePriorityEngine()
    >>> decision = engine.evaluate(machine_data)
    >>> engine.display(decision)
    """

    # ANSI colour codes -- priority -> colour escape
    _PRIORITY_COLOURS: Dict[str, str] = {
        "CRITICAL": "\033[91m",   # bright red
        "HIGH": "\033[93m",       # bright yellow
        "MEDIUM": "\033[94m",     # bright blue
        "LOW": "\033[92m",        # bright green
    }
    _RESET: str = "\033[0m"
    _BOLD: str = "\033[1m"

    # ------------------------------------------------------------------
    def evaluate(self, data: MachineData) -> MaintenanceDecision:
        """
        Evaluate a single machine record and return a maintenance decision.

        Processing steps
        ----------------
        1. Validate inputs.
        2. Derive base priority from failure probability.
        3. Identify active escalation flags.
        4. Escalate priority by one level if any flag is active.
        5. Assemble plain-English reasons and recommended actions.
        6. Return a fully populated :class:`MaintenanceDecision`.

        Parameters
        ----------
        data : MachineData
            Sensor readings and prediction for one machine.

        Returns
        -------
        MaintenanceDecision
            Fully enriched maintenance decision record.

        Raises
        ------
        TypeError
            If ``machine_id`` is not a non-empty string.
        ValueError
            If any input field fails the validation checks.
        """
        logger.info(
            "Evaluating machine: %s  (failure_prob=%.4f)",
            data.machine_id,
            data.failure_probability,
        )

        # Step 1: Validate ------------------------------------------------
        _validate_input(data)

        # Step 2: Base priority -------------------------------------------
        base_priority = _base_priority(data.failure_probability)
        logger.debug(
            "Base priority for %s: %s", data.machine_id, base_priority.value
        )

        # Step 3: Escalation flags ----------------------------------------
        escalation_flags = _build_escalation_flags(data)
        logger.debug(
            "Escalation flags for %s: %s", data.machine_id, escalation_flags
        )

        # Step 4: Escalate if warranted -----------------------------------
        final_priority = base_priority
        if escalation_flags:
            final_priority = _escalate(base_priority)
            if final_priority != base_priority:
                logger.info(
                    "Priority for %s escalated: %s -> %s  (flags: %s)",
                    data.machine_id,
                    base_priority.value,
                    final_priority.value,
                    escalation_flags,
                )

        # Step 5: Reasons and actions -------------------------------------
        reasons = _build_reasons(
            data, base_priority, escalation_flags, final_priority
        )
        actions = _build_actions(data, final_priority, escalation_flags)

        # Step 6: Assemble decision ---------------------------------------
        decision = MaintenanceDecision(
            machine_id=data.machine_id,
            priority=final_priority,
            maintenance_window=_MAINTENANCE_WINDOWS[final_priority],
            recommended_actions=actions,
            reason=reasons,
            failure_probability=data.failure_probability,
            machine_type=data.machine_type,
        )

        logger.info(
            "Decision for %s -- Priority: %s | Window: %s",
            data.machine_id,
            final_priority.value,
            decision.maintenance_window,
        )
        return decision

    # ------------------------------------------------------------------
    def evaluate_batch(
        self, records: List[MachineData]
    ) -> List[MaintenanceDecision]:
        """
        Evaluate a list of machine records and return decisions sorted by
        descending priority (CRITICAL first, LOW last).

        Parameters
        ----------
        records : list[MachineData]
            One :class:`MachineData` per machine.

        Returns
        -------
        list[MaintenanceDecision]
            Decisions sorted from highest to lowest priority.
        """
        if not records:
            logger.warning(
                "evaluate_batch called with an empty list -- returning []."
            )
            return []

        logger.info("Evaluating batch of %d machine(s).", len(records))
        decisions = [self.evaluate(r) for r in records]

        # Sort CRITICAL (3) -> HIGH (2) -> MEDIUM (1) -> LOW (0)
        decisions.sort(
            key=lambda d: _PRIORITY_ORDER[d.priority], reverse=True
        )
        return decisions

    # ------------------------------------------------------------------
    def display(self, decision: MaintenanceDecision) -> None:
        """
        Pretty-print a single :class:`MaintenanceDecision` to stdout.

        Output is structured and colour-coded by priority level using ANSI
        escape codes (works on any POSIX terminal or Windows Terminal).

        Parameters
        ----------
        decision : MaintenanceDecision
            The decision to display.
        """
        colour = self._PRIORITY_COLOURS.get(decision.priority.value, "")
        reset = self._RESET
        bold = self._BOLD

        machine_label = decision.machine_id
        if decision.machine_type:
            machine_label += f"  [{decision.machine_type}]"

        print()
        print(f"{bold}{'=' * 67}{reset}")
        print(f"{bold}  Machine  : {machine_label}{reset}")
        print(
            f"  Priority : {colour}{bold}{decision.priority.value:<10s}{reset}  "
            f"(Failure Probability: {decision.failure_probability * 100:.1f}%)"
        )
        print(f"  Window   : {bold}{decision.maintenance_window}{reset}")
        print(f"{'- ' * 33}{'-'}")

        print(f"{bold}  Reasons:{reset}")
        for idx, reason in enumerate(decision.reason, start=1):
            print(f"    {idx}. {reason}")

        print(f"{'- ' * 33}{'-'}")
        print(f"{bold}  Recommended Actions:{reset}")
        for idx, action in enumerate(decision.recommended_actions, start=1):
            print(f"    {idx}. {action}")

        print(f"{'=' * 67}")

    # ------------------------------------------------------------------
    def display_batch(self, decisions: List[MaintenanceDecision]) -> None:
        """
        Pretty-print a batch of decisions, preceded by a fleet summary table.

        Parameters
        ----------
        decisions : list[MaintenanceDecision]
            Pre-evaluated decisions (typically the output of
            :meth:`evaluate_batch`, which returns them sorted by priority).
        """
        if not decisions:
            print("No maintenance decisions to display.")
            return

        bold = self._BOLD
        reset = self._RESET

        # Fleet summary header
        print()
        print(f"{bold}{'=' * 67}{reset}")
        print(
            f"{bold}"
            f"{'MAINTENANCE PRIORITY ENGINE -- FLEET SUMMARY':^67}"
            f"{reset}"
        )
        print(f"{bold}{'=' * 67}{reset}")
        print()

        # Summary statistics
        priority_counts: Dict[str, int] = {p.value: 0 for p in Priority}
        for dec in decisions:
            priority_counts[dec.priority.value] += 1

        print(
            f"{bold}  Fleet Status  ({len(decisions)} machine(s) evaluated):{reset}"
        )
        for priority in [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.MEDIUM,
            Priority.LOW,
        ]:
            count = priority_counts[priority.value]
            colour = self._PRIORITY_COLOURS.get(priority.value, "")
            bar = "#" * count if count > 0 else "--"
            print(
                f"  {colour}{bold}{priority.value:<10s}{reset}  "
                f"{bar}  ({count})"
            )

        # Per-machine decisions
        print()
        print(
            f"{bold}  Detailed Decisions (sorted by descending priority):{reset}"
        )
        for decision in decisions:
            self.display(decision)


# ===========================================================================
# CONVENIENCE FUNCTION
# ===========================================================================


def evaluate_machine(
    machine_id: str,
    failure_probability: float,
    temperature: float,
    pressure: float,
    vibration: float,
    prediction_result: bool,
    humidity: Optional[float] = None,
    machine_type: Optional[str] = None,
    last_service_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Module-level convenience wrapper around :class:`MaintenancePriorityEngine`.

    Accepts individual sensor values instead of a :class:`MachineData` object
    and returns the decision as a plain dictionary (JSON-serialisable).
    Mirrors the ``run_inference`` convenience pattern used in the inference
    sub-package.

    Parameters
    ----------
    machine_id : str
        Unique machine identifier.
    failure_probability : float
        Predicted failure probability in ``[0.0, 1.0]``.
    temperature : float
        Temperature reading in degrees Celsius.
    pressure : float
        Pressure reading in bar.
    vibration : float
        Vibration reading in mm/s.
    prediction_result : bool
        Binary failure flag emitted by the ML model.
    humidity : float, optional
        Relative humidity in percent.
    machine_type : str, optional
        Human-readable machine category.
    last_service_days : int, optional
        Days elapsed since last maintenance service.

    Returns
    -------
    dict
        JSON-safe dictionary with the schema::

            {
                "machine_id"           : str,
                "priority"             : str,
                "maintenance_window"   : str,
                "recommended_actions"  : list[str],
                "reason"               : list[str],
                "failure_probability"  : float,
                "machine_type"         : str | None,
            }
    """
    data = MachineData(
        machine_id=machine_id,
        failure_probability=failure_probability,
        temperature=temperature,
        pressure=pressure,
        vibration=vibration,
        prediction_result=prediction_result,
        humidity=humidity,
        machine_type=machine_type,
        last_service_days=last_service_days,
    )
    engine = MaintenancePriorityEngine()
    decision = engine.evaluate(data)
    return decision.to_dict()


# ===========================================================================
# DEMO EXECUTION
# Run as:  python -m src.configs.business.maintenance_priority
# ===========================================================================


def _run_demo() -> None:
    """
    Demonstrate the engine with a realistic fleet spanning all four priority
    levels and several escalation scenarios.
    """
    import json
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = MaintenancePriorityEngine()

    # ------------------------------------------------------------------
    # Sample fleet - each record exercises different business rules
    # ------------------------------------------------------------------
    fleet: List[MachineData] = [
        # 1. CRITICAL base probability + high temperature (stays CRITICAL)
        MachineData(
            machine_id="CNC-007",
            machine_type="CNC Machine",
            failure_probability=0.95,
            temperature=91.3,
            pressure=7.4,
            vibration=6.1,
            humidity=None,
            last_service_days=45,
            prediction_result=True,
        ),
        # 2. HIGH base probability + very high vibration (escalates to CRITICAL)
        MachineData(
            machine_id="PUMP-012",
            machine_type="Hydraulic Pump",
            failure_probability=0.78,
            temperature=74.5,
            pressure=9.8,
            vibration=9.7,
            humidity=None,
            last_service_days=55,
            prediction_result=True,
        ),
        # 3. MEDIUM base + temperature + pressure + humidity anomalies
        #    (multi_anomaly + service_overdue -> escalates to HIGH)
        MachineData(
            machine_id="COMP-034",
            machine_type="Air Compressor",
            failure_probability=0.62,
            temperature=83.1,
            pressure=11.2,
            vibration=8.5,
            humidity=88.0,
            last_service_days=75,
            prediction_result=True,
        ),
        # 4. LOW base -- no escalation triggers (stays LOW)
        MachineData(
            machine_id="CONV-005",
            machine_type="Conveyor Belt Motor",
            failure_probability=0.21,
            temperature=55.0,
            pressure=4.5,
            vibration=3.2,
            humidity=60.0,
            last_service_days=20,
            prediction_result=False,
        ),
        # 5. MEDIUM base + service overdue only (escalates to HIGH)
        MachineData(
            machine_id="TURB-019",
            machine_type="Steam Turbine",
            failure_probability=0.55,
            temperature=69.0,
            pressure=6.3,
            vibration=5.8,
            humidity=None,
            last_service_days=68,
            prediction_result=False,
        ),
        # 6. HIGH base + no escalation triggers (stays HIGH)
        MachineData(
            machine_id="GENR-002",
            machine_type="Industrial Generator",
            failure_probability=0.82,
            temperature=72.0,
            pressure=5.5,
            vibration=4.0,
            humidity=55.0,
            last_service_days=30,
            prediction_result=True,
        ),
    ]

    print()
    print("=" * 69)
    print("  IoT PREDICTIVE MAINTENANCE -- MAINTENANCE PRIORITY ENGINE DEMO")
    print("=" * 69)

    decisions = engine.evaluate_batch(fleet)
    engine.display_batch(decisions)

    # Show JSON dict output for the top-priority machine
    print()
    print("  [JSON Dict Output -- Top Priority Machine]")
    print("-" * 69)
    top_dict = decisions[0].to_dict()
    print(json.dumps(top_dict, indent=4))
    print("-" * 69)


if __name__ == "__main__":
    _run_demo()
