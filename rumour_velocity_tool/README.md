# Rumour Velocity and Response Threshold Tool

A Python-based decision-support application that analyses how quickly online rumours spread, estimates their potential reach, evaluates the risk of amplification through public responses, and recommends whether to remain silent, monitor, or review a response.

## Project Overview

The **Rumour Velocity and Response Threshold Tool** is designed to help analysts understand the spread and potential impact of online claims using measurable indicators rather than relying only on total mention counts.

The application evaluates:

- **Rumour Velocity:** How quickly a claim is spreading over time.
- **Growth and Acceleration:** Whether the circulation rate is increasing or decreasing.
- **Estimated Reach:** The potential audience exposed to the claim.
- **Amplification Risk:** Whether a public response could introduce the claim to additional audiences.
- **Hardening Risk:** Whether prolonged silence could allow an unverified claim to become widely accepted.
- **Response Pressure Score (RPS):** A weighted score used to compare circulation indicators against configurable decision thresholds.

The tool generates one of four outcomes:

- `REMAIN SILENT`
- `MONITOR`
- `RECOMMEND RESPONSE REVIEW`
- `INSUFFICIENT DATA / HUMAN VERIFICATION REQUIRED`

A key feature is the **Silence Justification Engine**, which explains why remaining silent may be appropriate based on the available evidence and calculated metrics.

> Silence is not the absence of action. It is a deliberate, evidence-based decision supported by continuous monitoring and clearly defined intervention thresholds.

## Key Features

- Interactive Streamlit dashboard.
- Rumour and claim registration.
- Observation management and CSV uploads.
- Data validation and cleaning.
- Rumour velocity, growth and acceleration calculations.
- Reach and engagement analysis.
- Public-response amplification simulation.
- Response Pressure Score calculation.
- Configurable thresholds and sensitivity analysis.
- Automated decision explanations.
- Silence and hardening risk assessment.
- What-if simulation for response scenarios.
- Assessment history and PDF report generation.
- SQLite database for local data storage.
- Automated test suite.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly |
| Database | SQLite |
| PDF Generation | ReportLab |
| Testing | Pytest |
| Machine Learning / Text Analysis | Scikit-learn (TF-IDF) |

## System Architecture

The application follows a modular processing pipeline:

```text
Claim Input
    |
    v
Data Validation and Cleaning
    |
    v
Velocity Analysis
    |
    v
Growth and Acceleration Analysis
    |
    v
Reach Assessment
    |
    v
Amplification and Hardening Risk
    |
    v
Response Pressure Score (RPS)
    |
    v
Decision Engine
    |
    v
Recommendation and Justification
    |
    v
Monitoring, Reassessment and Reporting
```

## Mathematical Models

### 1. Rumour Velocity

Measures the rate at which mentions increase over time.

`V(t) = [M(t) - M(t-1)] / Δt`

Where:
- `M(t)` = Current cumulative mention count.
- `M(t-1)` = Previous cumulative mention count.
- `Δt` = Time difference in hours.

Unit: Mentions per hour.

### 2. Growth Rate

Measures the percentage change in mentions.

`G(t) = [(M(t) - M(t-1)) / M(t-1)] × 100`

### 3. Acceleration

Measures the change in rumour velocity over time.

`A(t) = [V(t) - V(t-1)] / Δt`

### 4. Response Pressure Score

The Response Pressure Score combines normalized indicators using configurable weights.

`RPS = w1 × Vn + w2 × Gn + w3 × An + w4 × Rn`

Where:
- `Vn` = Normalized velocity.
- `Gn` = Normalized growth.
- `An` = Normalized acceleration.
- `Rn` = Normalized reach.
- `w1, w2, w3, w4` = Configurable weights.

The weights are normalized when a component is unavailable, and the application identifies missing information instead of assuming values.

## Decision Framework

The decision engine evaluates the calculated metrics, configured thresholds and safety conditions.

| Condition | Recommendation |
|---|---|
| Low circulation and low risk | REMAIN SILENT |
| Increasing circulation or uncertain risk | MONITOR |
| Response threshold crossed | RECOMMEND RESPONSE REVIEW |
| Insufficient observations or evidence | INSUFFICIENT DATA / HUMAN VERIFICATION REQUIRED |

Safety-related concerns and serious credible allegations require human escalation regardless of the calculated score.

All recommendations are advisory. The application does not automatically publish public statements or responses.

## Dashboard Modules

1. **Executive Overview:** Displays key metrics, active claims and current recommendations.
2. **Rumour Input:** Register claims, add observations and upload CSV datasets.
3. **Velocity Analysis:** Visualizes mention counts, velocity, growth, acceleration and lifecycle stages.
4. **Amplification and Risk:** Evaluates estimated reach, engagement and potential response amplification.
5. **Response Threshold and Decision:** Displays RPS, configurable thresholds, decision explanations and sensitivity analysis.
6. **What-If Simulation:** Compares hypothetical outcomes of remaining silent versus issuing a public response.
7. **Assessment History and Reports:** Stores assessment history and generates PDF reports.

## Project Structure

```text
rumour_velocity_tool/
│
├── app.py
├── config.py
├── run_scenarios.py
├── requirements.txt
│
├── modules/
│   ├── data_processing.py
│   ├── velocity_engine.py
│   ├── reach_engine.py
│   ├── amplification_engine.py
│   ├── threshold_engine.py
│   ├── decision_engine.py
│   ├── silence_justification.py
│   ├── lifecycle_engine.py
│   ├── simulation_engine.py
│   ├── hardening_engine.py
│   └── pipeline.py
│
├── database/
│   ├── schema.sql
│   └── db_manager.py
│
├── reports/
│   └── report_generator.py
│
├── data/
│   ├── synthetic_rumours.csv
│   └── generate_synthetic.py
│
├── tests/
│
└── docs/
```

## Installation and Setup

### Prerequisites

- Python 3.11 or later
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Ravikumarkushwaha/Rumour-Velocity-and-Response-Threshold-Tool.git
```

### 2. Navigate to the Project Directory

```bash
cd Rumour-Velocity-and-Response-Threshold-Tool/rumour_velocity_tool
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

Run the predefined synthetic scenarios:

```bash
python run_scenarios.py
```

The project includes **146 automated tests** covering mathematical calculations, data validation, decision logic, risk assessment, database operations, simulations and PDF generation.

The predefined scenarios are regression tests using fictional data. They demonstrate expected application behaviour and do not establish real-world predictive accuracy.

## Data and Assumptions

The project uses synthetic, fictional datasets for demonstration and testing.

- Mention counts represent cumulative observations.
- Reach and amplification estimates depend on configurable assumptions.
- Source reliability is an analyst-assigned assessment, not machine-verified.
- Thresholds and weights are experimental and configurable.
- Missing information is not automatically treated as zero or fabricated.
- Public-source references should contain genuine public citations when available.

## Limitations

- The application does not independently verify whether a claim is true or false.
- Accuracy depends on the quality and completeness of input observations.
- Reach and amplification estimates rely on assumptions.
- Thresholds have not been empirically calibrated against a representative real-world dataset.
- Duplicate reposts within externally collected mention counts must be handled during data collection.
- The application is a decision-support prototype and requires human review.

## Future Enhancements

- Integration with permitted public data sources.
- Advanced NLP-based claim similarity and classification.
- Improved reach estimation methods.
- Historical trend analysis.
- Additional visualization and reporting capabilities.
- Evaluation using appropriately validated historical datasets.

## Author

**Ravi Kumar Kushwaha**

B.Tech – Computer Science and Engineering 

## Disclaimer

This project is an independent academic and technical prototype developed for research and demonstration purposes. It is not a source-verification system, an automated fact-checking authority or an autonomous public communication platform. All recommendations are advisory and require appropriate human assessment.
