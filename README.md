# 🩺 Smart Healthcare Assistant (VitalCare)

Educational prototype for **user-entered vital signs**: blood pressure, SpO₂, temperature, heart rate, optional glucose, and free-text symptoms.

**Not a medical device.** Does not diagnose, prescribe, or replace a clinician. Use only synthetic / demo data. Seek professional care for real symptoms.

## What's included

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (VitalCare dashboard) |
| `healthcare_core.py` | Shared backend: validation, thresholds, knowledge base, SQLite history, tests |
| `healthcare_mcp_server.py` | Optional MCP server (`analyze_vitals`, `save_vitals`, `get_vitals_history`, `search_health_knowledge`) |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd <repo-name>

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Optional: MCP server

```bash
python healthcare_mcp_server.py
```

Connect from an MCP-compatible host via stdio. Do **not** expose real patient data through an untrusted host.

## Features

- Pydantic-validated manual vital entry (HR, BP, SpO₂, temperature, glucose, symptoms)
- Shared threshold logic → transparent alerts (low / moderate / high / critical)
- Overall status: healthy · monitor · urgent · emergency
- Educational care suggestions + first-aid guidance when escalation is needed
- Local SQLite measurement history (demo only)
- 14 synthetic patients across common conditions for demos
- Automated backend tests: `python -c "from healthcare_core import run_tests; print(run_tests())"`

## Safety & privacy

- Educational prototype only — not FDA/CE cleared medical software.
- Measurements are self-reported and may be inaccurate.
- Pulse oximeter readings have known limitations; do not use them in isolation ([FDA guidance](https://www.fda.gov/medical-devices/products-and-medical-procedures/pulse-oximeters)).
- Do not enter identifiable real patient data. SQLite history is local to the runtime and is **not** production-grade PHI storage.

## Project layout

```
.
├── app.py                      # Streamlit entry point
├── healthcare_core.py          # Backend + tests
├── healthcare_mcp_server.py    # Optional MCP tools
├── requirements.txt
└── README.md
```

## License

Use freely for learning and demos. No warranty. Not for clinical decision-making.
