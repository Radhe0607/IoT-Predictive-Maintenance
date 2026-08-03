# IoT Predictive Maintenance

> **Enterprise-grade AI-driven predictive maintenance platform for industrial manufacturing.**
> Converts raw IoT sensor streams into structured failure predictions, maintenance priorities, financial impact estimates, and actionable business decisions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Configuration](#configuration)
- [Business Engines Demo](#business-engines-demo)
- [Technologies](#technologies)
- [Contributing](#contributing)

---

## Overview

Traditional predictive maintenance systems stop at "will this machine fail?"
This platform answers the questions that matter to operations teams:

| Question | Answered by |
|---|---|
| Will this machine fail? | ML Inference Pipeline |
| Which machine needs attention first? | Maintenance Priority Engine |
| How urgent is it? | Priority Level (CRITICAL / HIGH / MEDIUM / LOW) |
| What should we do? | Recommended Actions |
| How much will a failure cost? | Cost Estimation Engine |
| Is preventive maintenance worth it? | ROI Analysis |

---

## Architecture

`
Raw Sensor Data  (CSV / DataFrame / Dict)
        |
        v
+----------------------------------------------------------+
|                     Data Layer                            |
|  DataLoader --> DataPreprocessor --> FeatureEngineer     |
+----------------------------+-----------------------------+
                             |  Engineered Feature DataFrame
                             v
+----------------------------------------------------------+
|                     ML Pipeline                           |
|  BaselineModel (RandomForest) --> ModelEvaluator         |
|  EvaluationPipeline --> ReportGenerator                  |
+----------------------------+-----------------------------+
                             |  Trained Model Artifact (.joblib)
                             v
+----------------------------------------------------------+
|                  Inference Pipeline                       |
|  InferencePipeline --> InferenceResult                   |
|  (failure_probability, prediction, confidence)           |
+----------------------------+-----------------------------+
                             |  Prediction + Probability
                             v
+----------------------------------------------------------+
|                   Business Layer                          |
|                                                           |
|  MaintenancePriorityEngine                               |
|  +-- Priority Level  (CRITICAL / HIGH / MEDIUM / LOW)   |
|  +-- Maintenance Window  ("Within 24 Hours", ...)       |
|  +-- Recommended Actions                                 |
|  +-- Reasons                                             |
|                           |                              |
|                           v                              |
|  CostEstimationEngine                                    |
|  +-- Repair Cost  ($)                                   |
|  +-- Downtime Hours                                      |
|  +-- Production Loss  ($)                               |
|  +-- Preventive Maintenance Cost  ($)                   |
|  +-- Estimated Savings  ($)                             |
|  +-- ROI  (%)                                           |
|  +-- Business Risk  (VERY HIGH / HIGH / MEDIUM / LOW)  |
+----------------------------------------------------------+
`

### Pipeline Flow (Mermaid)

`mermaid
flowchart TD
    A[Raw Sensor CSV] --> B[DataLoader]
    B --> C[DataPreprocessor]
    C --> D[FeatureEngineer]
    D --> E[BaselineModel Training]
    E --> F[Model Artifact .joblib]
    F --> G[InferencePipeline]
    G --> H{Prediction + Probability}
    H --> I[MaintenancePriorityEngine]
    I --> J[Priority Decision]
    J --> K[CostEstimationEngine]
    K --> L[Financial Impact Report]
    L --> M[ReportGenerator]
    M --> N[JSON + TXT Reports]
`

---

## Folder Structure

`
IoT-Predictive-Maintenance/
|
+-- README.md
+-- requirements.txt
+-- .gitignore
|
+-- data/
    +-- notebook/
        +-- src/
            |
            +-- configs/                        # Project-wide configuration
            |   +-- config.py                   # Singleton config loader + frozen dataclasses
            |   +-- config.yaml                 # All tunable parameters (single source of truth)
            |   |
            |   +-- business/                   # Business intelligence layer
            |   |   +-- maintenance_priority.py # Priority Engine: CRITICAL/HIGH/MEDIUM/LOW
            |   |   +-- maintenance_cost.py     # Cost Engine: repair cost, downtime, ROI
            |   |
            |   +-- evaluation/                 # Model evaluation
            |   |   +-- evaluation_pipeline.py  # End-to-end evaluation orchestrator
            |   |   +-- model_evaluation.py     # Metrics, confusion matrix, ROC curve
            |   |   +-- eda.py                  # Exploratory data analysis plots
            |   |
            |   +-- features/
            |   |   +-- feature_engineering.py  # Rolling stats, lags, interactions, selection
            |   |
            |   +-- inference/
            |   |   +-- inference_pipeline.py   # Load model -> predict -> InferenceResult
            |   |
            |   +-- models/
            |   |   +-- baseline_model.py       # RandomForest training + evaluation plots
            |   |
            |   +-- reports/
            |   |   +-- report_generator.py     # JSON + TXT report generation
            |   |
            |   +-- utils/
            |       +-- model_manager.py        # joblib save / load with atomic writes
            |       +-- predict.py              # PredictionPipeline (backward-compat)
            |
            +-- data/
                +-- data_loader.py              # CSV / Parquet loader
                +-- preprocessing.py            # Imputation, encoding, normalisation
`

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Clone and install

`ash
git clone https://github.com/Radhe0607/IoT-Predictive-Maintenance.git
cd IoT-Predictive-Maintenance
pip install -r requirements.txt
`

### Conda environment (recommended)

`ash
conda create -n iot-pm python=3.11 -y
conda activate iot-pm
pip install -r requirements.txt
`

---

## Quick Start

All commands run from the data/notebook/ directory so that src is on the Python path.

### 1. Run inference

`python
from src.configs.inference import run_inference

result = run_inference("data/raw/sensor_data.csv")
result.display()
df = result.to_dataframe()
`

### 2. Run the Business Priority Engine

`python
from src.configs.business import MaintenancePriorityEngine, MachineData

engine = MaintenancePriorityEngine()

data = MachineData(
    machine_id="PUMP-012",
    machine_type="Hydraulic Pump",
    failure_probability=0.87,
    temperature=76.0,
    pressure=9.2,
    vibration=9.5,
    prediction_result=True,
    last_service_days=62,
)

decision = engine.evaluate(data)
engine.display(decision)
`

### 3. Run the Cost & Downtime Engine

`python
from src.configs.business import CostEstimationEngine, CostInput, Priority

engine = CostEstimationEngine()

inp = CostInput(
    machine_id="PUMP-012",
    machine_type="Hydraulic Pump",
    priority=Priority.CRITICAL,
    failure_probability=0.87,
    prediction_result=True,
    machine_age_years=5.0,
    operating_hours=11_000,
    last_service_days=62,
)

report = engine.estimate(inp)
engine.display_report(report)
print(report.to_dict())
`

### 4. Run the built-in demos

`ash
cd data/notebook

# Priority Engine -- 6-machine fleet demo
python -m src.configs.business.maintenance_priority

# Cost Engine -- 6-machine financial impact demo
python -m src.configs.business.maintenance_cost
`

---

## Module Reference

### Data Layer

| Module | Class | Purpose |
|---|---|---|
| data.data_loader | DataLoader | Load CSV / Parquet sensor datasets |
| data.preprocessing | DataPreprocessor | Imputation, encoding, normalisation |

### ML Pipeline

| Module | Class / Function | Purpose |
|---|---|---|
| configs.features.feature_engineering | FeatureEngineer | Rolling stats, lags, interactions, selection |
| configs.models.baseline_model | BaselineModel | RandomForest training + evaluation plots |
| configs.evaluation.evaluation_pipeline | EvaluationPipeline, 
un_evaluation | Post-training evaluation orchestrator |
| configs.evaluation.model_evaluation | ModelEvaluator | Accuracy, Precision, Recall, F1, ROC-AUC |
| configs.evaluation.eda | EDAAnalyser | Distribution plots, correlation matrix |
| configs.inference | InferencePipeline, 
un_inference | Load model, predict, return InferenceResult |
| configs.reports | ReportGenerator, generate_report | JSON + TXT report export |
| configs.utils | save_model, load_model | Atomic joblib model persistence |

### Business Layer

#### Maintenance Priority Engine

Converts failure probability + live sensor readings into a structured priority decision.

**Priority assignment:**

| Probability | Base Priority |
|---|---|
| >= 0.90 | CRITICAL |
| >= 0.75 | HIGH |
| >= 0.50 | MEDIUM |
| < 0.50  | LOW |

**Risk escalation** (bumps priority +1 level, capped at CRITICAL) when any condition holds:
- Temperature > 80 C
- Vibration > 8 mm/s
- Last service > 60 days
- Two or more sensors simultaneously abnormal
- ML model predicted failure with >= 80% confidence

**Output schema:**
`json
{
  "machine_id": "PUMP-012",
  "priority": "CRITICAL",
  "maintenance_window": "Within 24 Hours",
  "recommended_actions": ["..."],
  "reason": ["..."],
  "failure_probability": 0.87,
  "machine_type": "Hydraulic Pump"
}
`

#### Cost & Downtime Estimation Engine

Converts a priority decision into a fully itemised financial impact report.

**Repair cost formula:**
`
repair_cost = base_interpolation(failure_probability)
            x age_multiplier(years)        # +3% per year
            x hours_multiplier(hours)      # +10% above 10,000 hrs
            x service_multiplier(days)     # +8% when overdue
`

**Production loss:**
`
production_loss = downtime_hours
               x hourly_production_cost[machine_type]
               x criticality_multiplier[priority]
`

**ROI:**
`
ROI (%) = (estimated_savings / preventive_cost) x 100
`

**Output schema:**
`json
{
  "machine_id": "PUMP-012",
  "priority": "CRITICAL",
  "business_risk": "VERY HIGH",
  "repair_cost": 2185.40,
  "downtime_hours": 21.36,
  "production_loss": 10882.56,
  "total_failure_cost": 13067.96,
  "preventive_cost": 1200.00,
  "estimated_savings": 11867.96,
  "roi": "989%",
  "recommendations": ["..."],
  "cost_breakdown": {}
}
`

---

## Configuration

All tunable parameters live in a single file:
**data/notebook/src/configs/config.yaml**

`yaml
model:
  target_col:    "failure"
  test_size:     0.20
  n_estimators:  200
  class_weight:  "balanced"

evaluation:
  cv_folds:             5
  confidence_threshold: 0.50

feature_engineering:
  rolling_windows:       [3, 5, 10]
  lag_steps:             [1, 3, 5]
  variance_threshold:    0.01
  correlation_threshold: 0.95

logging:
  level:  "INFO"
  format: "%(asctime)s  [%(levelname)s]  %(name)s -- %(message)s"
`

Access configuration anywhere in the project:

`python
from src.configs import get_config, get_model_path

cfg = get_config()                # singleton -- reads YAML exactly once
model_path = get_model_path(cfg)  # returns absolute pathlib.Path
`

---

## Business Engines Demo

`
===================================================================
  Machine  : CNC-007  [CNC Machine]
  Priority : CRITICAL    (Failure Probability: 95.0%)
  Window   : Within 24 Hours
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Reasons:
    1. Failure probability is 95.0% (base classification: CRITICAL)
    2. Temperature 91.3 C exceeds the 80 C safety threshold
    3. ML model predicted FAILURE with 95.0% confidence
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Recommended Actions:
    1. Immediately halt machine operation and isolate the unit
    2. Notify maintenance supervisor and safety team without delay
    3. Inspect and clean cooling system (fans, heat exchangers, coolant levels)
===================================================================

============================================================
               Business Maintenance Cost Report
============================================================
  Machine:                     CNC-007  [CNC Machine]
  Priority:                    CRITICAL
  Failure Probability:         95.0%
  Business Risk:               VERY HIGH

  Financial Impact Summary
  ------------------------------------------------------------
  Estimated Repair Cost:              $     3,567
  Estimated Downtime:                 23.4 Hours
  Production Loss:                    $    15,795
  Total Failure Cost Exposure:        $    19,362
  Asset Replacement Value:            $    31,167

  Preventive Maintenance ROI Analysis
  ------------------------------------------------------------
  Preventive Maintenance Cost:        $     1,212
  Estimated Savings:                  $    18,150
  Return on Investment (ROI):              1498%
============================================================
`

---

## Technologies

| Technology | Version | Purpose |
|---|---|---|
| Python | >= 3.9 | Core language |
| pandas | >= 2.0 | Data manipulation |
| NumPy | >= 1.24 | Numerical operations |
| scikit-learn | >= 1.3 | ML model, preprocessing, metrics |
| scipy | >= 1.11 | Statistical utilities |
| joblib | >= 1.3 | Model serialisation |
| PyYAML | >= 6.0 | Configuration loading |
| matplotlib | >= 3.7 | Evaluation plots |
| seaborn | >= 0.13 | Plot aesthetics |
| pyarrow | >= 14.0 | Parquet I/O |

---

## Contributing

Please follow the conventions established throughout the codebase:

1. **Constants** belong in module-level UPPER_SNAKE_CASE blocks.
2. **Every public function and class** requires a NumPy-style docstring with
   Parameters, Returns, and Raises sections.
3. **Type hints** are required on every public function signature.
4. **Logging** via logging.getLogger(__name__) -- no bare print() in library code.
5. **Business logic** lives exclusively in src/configs/business/.

Quick syntax check before submitting a PR:

`ash
cd data/notebook
python -m py_compile src/configs/business/maintenance_priority.py
python -m py_compile src/configs/business/maintenance_cost.py
`

---

*Built during the Infotact Solutions Data Science & Machine Learning Internship.*
*Designed for enterprise Industrial AI, Manufacturing, and Predictive Maintenance platforms.*
