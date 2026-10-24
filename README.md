# GridGuard AI ⚡

**Intelligent Electricity Theft Detection and Smart Grid Anomaly Monitoring**

GridGuard AI is a software-centric electricity analytics platform engineered to detect non-technical losses (NTL / electricity theft), diagnose meter malfunctions, recognize communication dropouts, reconcile feeder/transformer energy balance, and continuously monitor smart-meter streams through an integrated virtual simulator.

---

## 📌 Executive Summary & Problem Domain

Electricity distribution utilities face immense non-technical losses (NTL) globally due to:
1. **Direct Tampering / Meter Bypass:** Diverting current before the meter or attaching bypass jumpers.
2. **Meter Slowdown / Hardware Tampering:** Applying physical magnets, altering neutral lines, or tampering with CT/PT ratios.
3. **Hardware Degradation & Faults:** Aging meter electronics causing drift or zero readings, distinct from intentional fraud.
4. **Communication Failures:** Telemetry packet loss in Advanced Metering Infrastructure (AMI) gateways misidentified as outages or theft.
5. **Legitimate Consumption Swings:** Rooftop solar injection, electric vehicle (EV) charging patterns, or prolonged consumer vacancy.

GridGuard AI addresses this by coupling **energy accounting** (transformer balance reconciliation) with **multi-class anomaly diagnostics** and **model explainability** to deliver actionable forensic insights for grid dispatchers and field crews.

---

## 📊 Phase 2: Real Smart-Meter Datasets & Provenance

The platform data foundation is powered by two real Advanced Metering Infrastructure (AMI) datasets:

### Dataset 1: `data/raw/data.csv` (SGCC Benchmark)
- **Origin:** State Grid Corporation of China (SGCC) public research benchmark for electricity theft detection.
- **Dimensions:** 42,372 rows × 1,036 columns (~167 MB).
- **Consumer Identifier:** `CONS_NO` — 32-character hexadecimal MD5 hash strings. 100% unique, 0 missing.
- **Candidate Label:** `FLAG` — Binary flag where `0` = normal / unflagged (38,757 consumers, 91.47%), `1` = flagged for non-technical loss (3,615 consumers, 8.53%).
- **Temporal Coverage:** 1,034 calendar days (`2014-01-01` to `2016-10-31`).
  - *Chronological Reordering:* The raw file columns are sorted lexically (e.g., `2014/1/1`, `2014/1/10`...); our pipeline parses and strictly sorts columns chronologically.
  - *Calendar Sequence Gap:* Exactly one calendar date is absent in the 1,035-day sequence: `2016-09-18`.
- **Measurement Characteristics:** 25.64% missing readings, 13.21% zero readings, **0 negative readings**.
- **Outliers:** Maximum recorded value reaches 800,003.32 with a 99.9th percentile of ~439 units.

### Dataset 2: `data/raw/Electricity_Theft_Data.csv` (2015 Cohort)
- **Origin:** Distribution utility smart-meter cohort for calendar year 2015.
- **Dimensions:** 9,957 rows × 367 columns (~14.4 MB).
- **Consumer Identifier:** `CONS_NO` — 9 to 10 digit numerical account identifier strings (e.g., `8546920085`).
- **Candidate Label:** `CHK_STATE` — Binary flag where `0` = normal (8,562 consumers, 86.00%), `1` = theft candidate (1,394 consumers, 14.00%), plus 1 row with NaN.
- **Temporal Coverage:** Exactly 365 days of calendar year 2015 (`2015-01-01` to `2015-12-31`). Header format `%d-%m-%y`.
- **Malformed Index Row 0 Artifact:**
  - Row 0 contains `CONS_NO: NaN`, `CHK_STATE: NaN`, and the 365 date columns contain sequential integers `1.0, 2.0, 3.0, ..., 365.0`.
  - **Forensic Diagnosis:** This is an exported column-index / day-of-year artifact from legacy spreadsheet processing.
  - **Pipeline Handling:** Row 0 is explicitly quarantined in `data/processed/quarantined_records.csv` and excluded from ML feature matrices.
- **Measurement Characteristics:** 13.58% missing readings, 15.15% zero readings, **1,442 negative readings** (minimum: `-82,809.93`).

### ⚠️ Known Provenance & Semantic Limitations
1. **Physical Units Undocumented:** Neither dataset specifies whether values are kWh, MWh, or meter pulses. The pipeline enforces a unit-agnostic designation `consumption_value`.
2. **Consumer-Level Labels vs. Daily Events:** Both `FLAG` and `CHK_STATE` are static labels assigned to the entire consumer record over multi-month windows. They do not indicate the exact day or hour theft commenced.
3. **Daily Aggregation Only:** Neither dataset contains sub-daily 15-minute load curves, active/reactive power, power factor, or voltage telemetry.
4. **Mutually Exclusive Namespaces:** The two datasets have **0 overlapping consumers**. They originate from different administrative metering systems.

---

## ⚖️ Cross-Dataset Compatibility & Architectural Strategy

As established in [`reports/dataset_compatibility_report.md`](file:///c:/Users/hp/Desktop/GridGuard%20AI/reports/dataset_compatibility_report.md):

> [!IMPORTANT]
> **Architectural Recommendation: KEEP STRICTLY SEPARATE.**
> Do not concatenate the two raw datasets into a single pooled training matrix.
> 
> - **Primary Training Cohort:** Use `data.csv` (42,372 meters over ~3 years) for primary baseline model training, time-series feature engineering, and cross-validation.
> - **Out-of-Distribution (OOD) Evaluation Cohort:** Use `Electricity_Theft_Data.csv` (9,956 meters) as an independent zero-shot evaluation benchmark to verify model transferability across distinct utility environments.

---

## 🛠️ Data Preprocessing & Validation Pipeline

The pipeline enforces rigorous data hygiene without silent data destruction:

1. **Chronological Sorting:** Replaces lexical string sorting with ISO date mapping (`YYYY-MM-DD`).
2. **Zero vs. Missing Distinction:**
   - Missing readings (`NaN`) are preserved as `None` with quality flag `MISSING_VALUE`. **They are never silently imputed as 0.0.**
   - Genuine `0.0` readings are preserved with quality flag `ZERO_READING` to differentiate seasonal vacancies or disconnections from communication dropouts.
3. **Negative Reading Tagging:** Flagged with `NEGATIVE_READING` rather than clipped to zero, maintaining physical evidence of reverse CT wiring or unmetered solar injection.
4. **Outlier Flagging:** Values exceeding 1,000 units are tagged with `OUTLIER_HIGH` for engineer inspection.
5. **Traceable Quarantine:** Every excluded or malformed row is logged to `data/processed/quarantined_records.csv` with a forensic reason.
6. **High-Performance Memory Management:** Chunked and vectorized NumPy/Pandas processing (`pd.melt`) ensures the 175 MB dataset processes in seconds with minimal RAM overhead.
7. **Indexed SQLite Persistence:** Ingests standardized consumption records and consumer labels into `database/gridguard.db` with B-tree indices on `consumer_id` and `timestamp`.

### Standardized Output Schemas

#### 1. Standardized Consumption Record (`standardized_consumption`)
| Column | Type | Description |
|---|---|---|
| `consumer_id` | `VARCHAR` | Unique consumer identifier string |
| `timestamp` | `VARCHAR` | ISO measurement date (`YYYY-MM-DD`) |
| `consumption_value` | `FLOAT` | Energy consumption measurement (null if missing) |
| `source_dataset` | `VARCHAR` | Originating raw CSV filename |
| `source_label` | `FLOAT` | Original raw label associated with consumer |
| `source_label_column` | `VARCHAR` | Original label column name (`FLAG`, `CHK_STATE`) |
| `data_quality_flags` | `VARCHAR` | Quality flags (`VALID`, `ZERO_READING`, `NEGATIVE_READING`, `MISSING_VALUE`, `OUTLIER_HIGH`) |

#### 2. Consumer Label Registry (`consumer_labels`)
| Column | Type | Description |
|---|---|---|
| `consumer_id` | `VARCHAR` | Unique consumer identifier string |
| `source_dataset` | `VARCHAR` | Source dataset (`data.csv` or `Electricity_Theft_Data.csv`) |
| `source_label_column` | `VARCHAR` | `FLAG` or `CHK_STATE` |
| `source_label` | `FLOAT` | `0.0` (normal) or `1.0` (flagged candidate) |
| `total_days` | `INT` | Total calendar days tracked |
| `recorded_days` | `INT` | Days with non-null measurements |
| `missing_days` | `INT` | Count of absent readings |
| `zero_days` | `INT` | Count of 0.0 readings |
| `negative_days` | `INT` | Count of negative readings |
| `mean_consumption` | `FLOAT` | Mean daily consumption |
| `max_consumption` | `FLOAT` | Maximum daily consumption |

---

## 🏗️ Repository Architecture

```
GridGuard AI/
├── app.py                     # Streamlit web dashboard with interactive Data Explorer
├── requirements.txt           # Core project dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusion rules
├── README.md                  # System documentation & setup guide
├── config/
│   ├── __init__.py
│   └── settings.py            # Centralized paths and settings dataclass
├── data/
│   ├── raw/
│   │   ├── data.csv                   # Raw SGCC dataset (175 MB)
│   │   └── Electricity_Theft_Data.csv # Raw 2015 dataset (15 MB)
│   ├── processed/
│   │   ├── dataset_inspection.json    # Inspection metadata summary
│   │   ├── dataset_compatibility.json # Cross-dataset compatibility metrics
│   │   ├── data_quality_report.json   # Validation transformation audit
│   │   ├── consumer_labels.csv        # 52,328 standardized consumer records
│   │   ├── quarantined_records.csv    # Traceable audit log of excluded records
│   │   └── standardized_consumption_sample.csv # 1.39M long-format readings
│   └── synthetic/             # Future simulated tamper scenarios
├── database/
│   └── gridguard.db           # Indexed SQLite database with standardized tables
├── models/                    # Serialized ML model registry
├── reports/
│   ├── dataset_inspection_report.md     # Markdown inspection report
│   ├── data_quality_report.md           # Markdown data quality report
│   └── dataset_compatibility_report.md  # Markdown cross-dataset compatibility report
├── logs/                      # Runtime application logs
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Standardized dataclasses and quality flags
│   │   ├── dataset_inspector.py # Chronological parsing, gap detection, profiling
│   │   ├── validation.py      # ID, label, timestamp, and profile validators
│   │   ├── adapters.py        # Wide-to-long transformation and pipeline runner
│   │   └── compatibility.py   # Cross-dataset comparative assessment
│   ├── features/              # Feature engineering (Phase 3)
│   ├── detection/             # Anomaly scoring & ML detectors (Phase 3)
│   ├── classification/        # Root-cause diagnostics (Phase 3)
│   ├── explainability/        # SHAP attribution & field summaries (Phase 4)
│   ├── energy_accounting/     # Feeder balance reconciliation (Phase 4)
│   ├── simulator/             # Virtual smart-meter telemetry generator
│   └── services/              # Persistence and backend integration
└── tests/
    ├── __init__.py
    ├── test_app.py            # Configuration and initialization tests
    ├── test_dataset_inspector.py # Date sorting, gap detection, malformed row tests
    ├── test_validation.py     # Domain validation rules and quality flags tests
    ├── test_adapters.py       # Wide-to-long conversion, hygiene, real data slice
    └── test_compatibility.py  # Compatibility assessment tests
```

---

## 🚀 Execution Guide (Windows PowerShell)

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Run Data Inspection Pipeline
Inspects raw datasets, detects date sequence gaps, checks malformed rows, and generates JSON + Markdown reports:
```powershell
python src/data/dataset_inspector.py
```

### 3. Run Ingestion, Validation & SQLite Pipeline
Extracts consumer labels, quarantines malformed records, converts wide-to-long, tags quality flags, and indexes SQLite storage:
```powershell
python src/data/adapters.py
```

### 4. Run Cross-Dataset Compatibility Assessment
Evaluates identifier formats, temporal overlap, and generates compatibility reports:
```powershell
python src/data/compatibility.py
```

### 5. Run the Full Automated Test Suite
Runs all 23 unit, validation, adapter, compatibility, and real-data integration tests:
```powershell
python -m unittest discover tests -v
```
*(Or with Pytest if installed: `pytest -v`)*

### 6. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`. Navigate between:
- **📊 Data Explorer & Pipeline:** Interactive inspection, label distribution charts, individual smart-meter time-series curves, compatibility metrics, and quarantine audit logs.
- **🏛️ Architecture & Roadmap:** Architectural pillars and engineering roadmap.

---

## 🗺️ Phased Development Roadmap

| Phase | Title | Scope & Deliverables | Status |
|---|---|---|---|
| **Phase 1** | **Project Initialization** | Directory structure, settings, `.gitignore`, `.env.example`, initial tests | **Completed** |
| **Phase 2** | **Data Foundation & Hygiene** | Ingestion pipeline, chronological date parsing, wide-to-long adapters, quarantine audit, SQLite indexing, Data Explorer dashboard | **Completed** |
| **Phase 3** | **Feature Engineering & ML Detection** | Rolling consumption variance, Peak-to-Average Ratio (PAR), zero-reading streaks, Isolation Forests, supervised theft classifier, cause disambiguation | Staged Next |
| **Phase 4** | **Explainability & Accounting** | SHAP explanations, field inspection notes, feeder/DT energy balance engine | Planned |
| **Phase 5** | **Backend API & Streaming Ops** | Virtual smart-meter simulator, FastAPI endpoints, continuous alerting | Planned |
