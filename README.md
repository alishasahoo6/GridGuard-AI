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

## 🏗️ Architectural Overview & Modules

The platform is structured cleanly to enforce separation of concerns across analytics, machine learning, simulation, and presentation:

```
gridguard-ai/
├── app.py                     # Streamlit web dashboard entrypoint
├── requirements.txt           # Core project dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusion rules
├── README.md                  # System documentation & setup guide
├── config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py            # Centralized paths and settings dataclass
├── data/                      # Data storage layers
│   ├── raw/                   # Raw meter logs & utility exports
│   ├── processed/             # Cleaned & transformed analytical matrices
│   └── synthetic/             # Generated consumption & tamper scenarios
├── models/                    # Serialized machine learning models
├── reports/                   # Generated audit reports and evaluation plots
├── logs/                      # Runtime application logs
├── database/                  # Local SQLite database files
├── src/                       # Core analytical and engine source code
│   ├── __init__.py
│   ├── data/                  # Ingestion, validation, and data synthesis
│   ├── features/              # Statistical signals & tamper feature extraction
│   ├── detection/             # Unsupervised & supervised theft detection models
│   ├── classification/        # Root-cause classification (Theft vs Fault vs Comm)
│   ├── explainability/        # SHAP attribution & field technician explanations
│   ├── energy_accounting/     # Feeder & distribution transformer reconciliation
│   ├── simulator/             # Virtual smart-meter stream generator
│   └── services/              # Persistence, notification, and API services
└── tests/                     # Automated test suite
    ├── __init__.py
    └── test_app.py            # Baseline initialization and configuration tests
```

### Module Responsibilities:
- **`config/settings.py`**: Immutable configuration dataclass providing path resolution across operating systems, environment variable loading, and operational defaults.
- **`src/data/`**: Ingestion pipelines for AMI time series and generation of ground-truth synthetic consumption profiles.
- **`src/features/`**: Feature transformations including Peak-to-Average Ratio (PAR), rolling consumption variance, zero-reading streaks, and weather correlations.
- **`src/detection/`**: Anomaly detection utilizing Isolation Forests, Autoencoders, and supervised classifiers.
- **`src/classification/`**: Cause disambiguation model distinguishing illegal abstraction from hardware degradation or communication loss.
- **`src/explainability/`**: Explainable AI (XAI) engine generating plain-language evidence for utility field audits.
- **`src/energy_accounting/`**: Comparison of upstream distribution transformer (DT) energy output with downstream aggregate meter consumption.
- **`src/simulator/`**: Virtual AMI simulator emitting realistic synthetic telemetry streams.
- **`src/services/`**: SQLite query operations, background tasks, and FastAPI service endpoints.

---

## 🚀 Setup & Installation (Windows)

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell or Command Prompt

### Step 1: Create a Virtual Environment

Open PowerShell inside the project directory:

```powershell
python -m venv venv
```

### Step 2: Activate the Virtual Environment

On Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

On Windows Command Prompt (`cmd.exe`):
```cmd
.\venv\Scripts\activate.bat
```

### Step 3: Install Core Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Configure Local Environment

Copy the example environment template to create your local `.env`:

```powershell
Copy-Item .env.example .env
```
*(On Command Prompt, use `copy .env.example .env`)*

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:

```powershell
pytest -v
```

If testing in an environment prior to installing `pytest`, run tests via Python's built-in `unittest` runner:

```powershell
python -m unittest discover tests -v
```

---

## 🖥️ Launching the Dashboard

Start the Streamlit application:

```powershell
streamlit run app.py
```

Once started, open your web browser to:
```
http://localhost:8501
```

### What You Will See in Phase 1:
- **System Header:** Project title, tagline, and active release version (`0.1.0-alpha`).
- **Initialization Status Banner:** Confirmation that directory structure, configurations, and core modules are initialized.
- **Platform Pillars:** Conceptual overview of theft detection, cause disambiguation, grid energy accounting, and virtual simulation.
- **Module Architecture Matrix:** Real status tracking of upcoming analytical modules (no fabricated detection metrics or imaginary alerts).
- **Workspace Verification:** Active paths and next development steps.

---

## 🗺️ Phased Development Roadmap

| Phase | Title | Scope & Deliverables | Status |
|---|---|---|---|
| **Phase 1** | **Project Initialization** | Directory structure, settings, `.gitignore`, `.env.example`, tests, minimal dashboard | **Completed** |
| **Phase 2** | **Data Synthesis & Simulator** | Synthetic load generator, tamper injection, virtual AMI telemetry simulator | Planned |
| **Phase 3** | **ML Detection & Root-Cause** | Unsupervised anomaly scoring, supervised theft classifier, cause disambiguation | Planned |
| **Phase 4** | **Explainability & Accounting** | SHAP explanations, field inspection reports, feeder/DT energy balance engine | Planned |
| **Phase 5** | **Backend API & Real-Time Ops** | FastAPI telemetry ingestion endpoints, SQLite persistence, continuous alerts | Planned |

