"""
src/configs/business/__init__.py
================================
Public interface for the IoT Predictive Maintenance **business logic** layer.

This package contains two engines that sit downstream of the ML inference
pipeline and translate raw failure predictions into decisions that operations
managers, maintenance engineers, and financial analysts can act on directly:

Pipeline position::

    ML Inference  (src.configs.inference)
         |
         v
    MaintenancePriorityEngine          <-- maintenance_priority.py
    Assigns: Priority level, maintenance window, action plan, reasons
         |
         v
    CostEstimationEngine               <-- maintenance_cost.py
    Assigns: Repair cost, downtime, production loss, ROI, business risk

Exported symbols
----------------
From ``maintenance_priority``:

    MaintenancePriorityEngine
        Stateless engine that converts sensor readings and failure
        probabilities into structured priority decisions
        (CRITICAL / HIGH / MEDIUM / LOW).

    MachineData
        Typed input dataclass for the priority engine.

    MaintenanceDecision
        Typed output dataclass with ``.to_dict()`` for JSON serialisation.

    Priority
        Enum for CRITICAL / HIGH / MEDIUM / LOW, with ordering support.

    evaluate_machine
        Convenience wrapper that accepts flat keyword arguments and returns
        a JSON-safe dict -- mirrors the ``run_inference`` pattern.

From ``maintenance_cost``:

    CostEstimationEngine
        Stateless engine that converts a priority decision into financial
        impact figures: repair cost, downtime hours, production loss,
        preventive maintenance cost, estimated savings, and ROI.

    CostInput
        Typed input dataclass for the cost engine.

    CostReport
        Typed output dataclass with ``.to_dict()`` and a ``.roi_label``
        property.  Carries the full itemised cost breakdown.

    BusinessRisk
        Enum for VERY HIGH / HIGH / MEDIUM / LOW risk classification.

    estimate_costs
        Convenience wrapper that accepts flat keyword arguments and returns
        a JSON-safe dict.

Usage::

    from src.configs.business import (
        MaintenancePriorityEngine, MachineData,
        CostEstimationEngine, CostInput,
        Priority,
    )

    priority_engine = MaintenancePriorityEngine()
    cost_engine     = CostEstimationEngine()

    machine = MachineData(
        machine_id="PUMP-012",
        machine_type="Hydraulic Pump",
        failure_probability=0.87,
        temperature=76.0,
        pressure=9.2,
        vibration=9.5,
        prediction_result=True,
        last_service_days=62,
    )

    decision = priority_engine.evaluate(machine)
    priority_engine.display(decision)

    cost_inp = CostInput(
        machine_id=machine.machine_id,
        machine_type=machine.machine_type,
        priority=decision.priority,
        failure_probability=machine.failure_probability,
        prediction_result=machine.prediction_result,
        machine_age_years=5.0,
        operating_hours=11_000,
        last_service_days=machine.last_service_days,
    )

    report = cost_engine.estimate(cost_inp)
    cost_engine.display_report(report)
"""

from .maintenance_priority import (
    MaintenancePriorityEngine,
    MachineData,
    MaintenanceDecision,
    Priority,
    evaluate_machine,
)

from .maintenance_cost import (
    CostEstimationEngine,
    CostInput,
    CostReport,
    BusinessRisk,
    estimate_costs,
)

__all__ = [
    "MaintenancePriorityEngine",
    "MachineData",
    "MaintenanceDecision",
    "Priority",
    "evaluate_machine",
    "CostEstimationEngine",
    "CostInput",
    "CostReport",
    "BusinessRisk",
    "estimate_costs",
]
