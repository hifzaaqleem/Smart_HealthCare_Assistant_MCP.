import streamlit as st
import pandas as pd
import html as html_lib

from healthcare_core import (
    run_full_assessment,
    get_measurement_history,
    save_measurement,
    patients,
    vitals,
    parse_bp,
)

# ------------------------------------------------------------
# Page config (must be first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Theme CSS (compact — no 4-space indent so markdown never treats HTML as code)
# ------------------------------------------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.stApp { background: #0b0b0d; color: #ffffff; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

.vc-header {
  background: #111113;
  border: 1px solid #1f1f23;
  border-radius: 14px;
  padding: 16px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 18px;
}
.vc-header-left { display: flex; align-items: center; gap: 12px; }
.vc-icon {
  background: #116124;
  color: #e2fceb;
  padding: 10px 12px;
  border-radius: 12px;
  font-size: 1.15rem;
  font-weight: 700;
}
.vc-name { font-size: 1.25rem; font-weight: 700; color: #fff; margin: 0; }
.vc-sub { font-size: 0.8rem; color: #9ca3af; margin: 2px 0 0 0; }
.vc-badge {
  background: #222228;
  color: #fb923c;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid #2f2f37;
}

.vc-card {
  background: #161619;
  border: 1px solid #222228;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  min-height: 130px;
  margin-bottom: 12px;
}
.vc-card-title { font-size: 0.85rem; color: #9ca3af; margin-bottom: 8px; }
.vc-bp { font-size: 2.1rem; font-weight: 700; color: #f87171; line-height: 1.1; }
.vc-spo2 { font-size: 2.1rem; font-weight: 700; color: #38bdf8; line-height: 1.1; }
.vc-temp { font-size: 2.1rem; font-weight: 700; color: #fb923c; line-height: 1.1; }
.vc-hr { font-size: 2.1rem; font-weight: 700; color: #c084fc; line-height: 1.1; }
.vc-glucose { font-size: 2.1rem; font-weight: 700; color: #34d399; line-height: 1.1; }
.vc-card-sub { color: #9ca3af; font-size: 0.78rem; margin-top: 6px; }

.alert-card {
  background: #161619;
  border: 1px solid #222228;
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 10px;
}
.alert-critical { border-left: 5px solid #dc2626; }
.alert-high { border-left: 5px solid #ea580c; }
.alert-moderate { border-left: 5px solid #d97706; }
.alert-low { border-left: 5px solid #2563eb; }

.vc-emergency {
  background: #1c1113;
  border: 1px solid #7f1d1d;
  border-radius: 14px;
  padding: 18px;
  margin-top: 12px;
}
.emergency-title { color: #f87171; font-size: 0.95rem; font-weight: 700; }
.emergency-sub { color: #fca5a5; font-size: 0.82rem; margin-top: 6px; }

.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: #111113 !important;
  color: #ffffff !important;
  border: 1px solid #2f2f37 !important;
  border-radius: 8px !important;
}
label { color: #d1d5db !important; }
.stButton > button {
  background: #166534 !important;
  color: white !important;
  border: none !important;
  border-radius: 9px !important;
  font-weight: 600 !important;
  min-height: 46px !important;
}
.stButton > button:hover { background: #15803d !important; }
.stTabs [data-baseweb="tab-list"] {
  background: #111113; border-radius: 10px; padding: 5px; gap: 5px;
}
.stTabs [data-baseweb="tab"] { color: #9ca3af; }
.stTabs [aria-selected="true"] { color: #ffffff !important; }
[data-testid="stDataFrame"] { border: 1px solid #222228; border-radius: 10px; }

.vc-footer {
  color: #6b7280;
  font-size: 0.78rem;
  margin-top: 8px;
  margin-bottom: 12px;
  line-height: 1.4;
}
.status-line {
  color: #9ca3af;
  font-size: 0.85rem;
  margin-bottom: 14px;
}
.patient-chip {
  background: #161619;
  border: 1px solid #2f2f37;
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 10px;
  font-size: 0.82rem;
  color: #d1d5db;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def fmt(value, decimals=1, suffix=""):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def vital_card(title: str, value: str, subtitle: str, css_class: str) -> str:
    return (
        f'<div class="vc-card">'
        f'<div class="vc-card-title">{title}</div>'
        f'<div class="{css_class}">{value}</div>'
        f'<div class="vc-card-sub">{subtitle}</div>'
        f"</div>"
    )


def severity_class(severity: str) -> str:
    s = str(severity).lower()
    if s == "critical":
        return "alert-critical"
    if s == "high":
        return "alert-high"
    if s == "moderate":
        return "alert-moderate"
    return "alert-low"


def render_html(snippet: str) -> None:
    if hasattr(st, "html"):
        st.html(snippet)
    else:
        st.markdown(snippet, unsafe_allow_html=True)


def build_patient_options():
    """Label → patient_id for the preset dropdown."""
    options = {"— Manual entry (type your own values) —": None}
    for _, row in patients.iterrows():
        label = f"#{row['patient_id']} · {row['name']} · {row['age']}y · {row['condition']}"
        options[label] = int(row["patient_id"])
    return options


def load_preset_defaults(patient_id: int) -> dict:
    """Return form defaults from synthetic patient + vitals tables."""
    prow = patients.loc[patients.patient_id == patient_id]
    vrow = vitals.loc[vitals.patient_id == patient_id]
    if prow.empty or vrow.empty:
        return {}
    p = prow.iloc[0]
    v = vrow.iloc[0]
    bp = parse_bp(v["blood_pressure"])
    sys_bp, dia_bp = bp if bp else (120, 80)
    return {
        "patient_id": int(p["patient_id"]),
        "age": int(p["age"]),
        "heart_rate": float(v["heart_rate"]),
        "glucose": float(v["glucose_level"]),
        "systolic_bp": int(sys_bp),
        "diastolic_bp": int(dia_bp),
        "spo2": float(v["spo2"]),
        "temperature": float(v["temperature_c"]),
        "known_condition": str(p["condition"]),
        "name": str(p["name"]),
    }


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
render_html(
    '<div class="vc-header">'
    '<div class="vc-header-left">'
    '<div class="vc-icon">🩺</div>'
    "<div>"
    '<p class="vc-name">Smart Healthcare Assistant</p>'
    '<p class="vc-sub">Educational vitals analysis · synthetic demo data</p>'
    "</div>"
    "</div>"
    '<div class="vc-badge">Demo</div>'
    "</div>"
)

# ------------------------------------------------------------
# Build patient option map once
# ------------------------------------------------------------
PATIENT_OPTIONS = build_patient_options()

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
left, right = st.columns([1, 2], gap="large")

# -------------------- LEFT: inputs --------------------
with left:
    st.subheader("⚙️ Patient Vitals Input")

    selected_label = st.selectbox(
        "Data source",
        options=list(PATIENT_OPTIONS.keys()),
        help="Pick a synthetic demo patient to auto-fill vitals, or use manual entry.",
    )
    selected_pid = PATIENT_OPTIONS[selected_label]

    def _apply_form_values(
        *,
        pid=0,
        age_v=33,
        hr=72.0,
        glucose_v=95.0,
        sys=120,
        dia=80,
        spo2_v=98.0,
        temp=36.8,
        condition="",
        symptoms_v="",
    ):
        # Set widget keys BEFORE widgets are created (Streamlit pattern)
        st.session_state["widget_patient_id"] = int(pid)
        st.session_state["widget_age"] = int(age_v)
        st.session_state["widget_hr"] = float(hr)
        st.session_state["widget_glucose"] = float(glucose_v)
        st.session_state["widget_sys"] = int(sys)
        st.session_state["widget_dia"] = int(dia)
        st.session_state["widget_spo2"] = float(spo2_v)
        st.session_state["widget_temp"] = float(temp)
        st.session_state["widget_condition"] = condition or ""
        st.session_state["widget_symptoms"] = symptoms_v or ""

    # First visit: seed defaults once
    if "widget_patient_id" not in st.session_state:
        _apply_form_values()

    preset = None
    if selected_pid is not None:
        preset = load_preset_defaults(selected_pid)
        if st.session_state.get("_last_preset_pid") != selected_pid:
            st.session_state["_last_preset_pid"] = selected_pid
            _apply_form_values(
                pid=preset.get("patient_id", 0),
                age_v=preset.get("age", 33),
                hr=preset.get("heart_rate", 72.0),
                glucose_v=preset.get("glucose", 95.0),
                sys=preset.get("systolic_bp", 120),
                dia=preset.get("diastolic_bp", 80),
                spo2_v=preset.get("spo2", 98.0),
                temp=preset.get("temperature", 36.8),
                condition=preset.get("known_condition", ""),
                symptoms_v="",
            )
        render_html(
            f'<div class="patient-chip">'
            f"<b>{html_lib.escape(preset.get('name', ''))}</b> · "
            f"ID {preset.get('patient_id')} · {preset.get('age')}y<br>"
            f"{html_lib.escape(preset.get('known_condition', ''))}"
            f"</div>"
        )
    else:
        if st.session_state.get("_last_preset_pid") is not None:
            st.session_state["_last_preset_pid"] = None
            _apply_form_values()

    patient_id = st.number_input(
        "Patient ID (optional)", min_value=0, step=1, key="widget_patient_id"
    )
    age = st.number_input(
        "Age", min_value=0, max_value=120, step=1, key="widget_age"
    )

    c1, c2 = st.columns(2)
    with c1:
        systolic_bp = st.number_input(
            "Systolic BP", min_value=0, max_value=260, step=1, key="widget_sys"
        )
    with c2:
        diastolic_bp = st.number_input(
            "Diastolic BP", min_value=0, max_value=180, step=1, key="widget_dia"
        )

    c1, c2 = st.columns(2)
    with c1:
        heart_rate = st.number_input(
            "Heart Rate (bpm)", min_value=0.0, max_value=250.0, step=1.0, key="widget_hr"
        )
    with c2:
        spo2 = st.number_input(
            "SpO₂ (%)", min_value=0.0, max_value=100.0, step=1.0, key="widget_spo2"
        )

    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input(
            "Temp (°C)", min_value=0.0, max_value=43.0, step=0.1, key="widget_temp"
        )
    with c2:
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0.0, max_value=700.0, step=1.0, key="widget_glucose"
        )

    symptoms = st.text_area(
        "Symptoms", placeholder="Describe symptoms…", height=90, key="widget_symptoms"
    )
    known_condition = st.text_input("Known condition", key="widget_condition")

    analyze = st.button("🔍 Analyze Vitals", use_container_width=True)

    st.markdown(
        '<p class="vc-footer">Select a <b>synthetic patient</b> above to auto-fill, '
        "or type values manually. Educational prototype only — not a medical device.</p>",
        unsafe_allow_html=True,
    )

# -------------------- RIGHT: dashboard --------------------
with right:
    if "assessment" not in st.session_state:
        display_bp, display_spo2, display_temp = "120/80", "98%", "36.8°C"
        display_hr, display_glucose = "72 BPM", "95 mg/dL"
        bp_label = spo2_label = temp_label = hr_label = glucose_label = "Example"
        alerts, recommendations, first_aid = [], [], {}
        status_text, status_label = "READY", "Waiting for analysis"
    else:
        assessment = st.session_state["assessment"]
        vitals_result = assessment.get("vitals", {})

        if vitals_result.get("status") != "ok":
            st.error(
                vitals_result.get("data", {}).get("message")
                or vitals_result.get("data", {}).get("validation_error")
                or "Unable to analyze the entered data."
            )
            st.stop()

        data = vitals_result.get("data", {})
        classification = data.get("classification", {})
        alerts = vitals_result.get("alerts", [])

        care_result = assessment.get("care") or {}
        care = care_result.get("data", care_result) if isinstance(care_result, dict) else {}
        recommendations = care.get("suggestions", [])
        first_aid = care.get("first_aid", {}) or {}

        if data.get("systolic_bp") is not None and data.get("diastolic_bp") is not None:
            display_bp = f"{data['systolic_bp']}/{data['diastolic_bp']}"
        else:
            display_bp = "—"

        display_spo2 = fmt(data.get("spo2"), 0, "%") if data.get("spo2") is not None else "—"
        display_temp = fmt(data.get("temperature_c"), 1, "°C") if data.get("temperature_c") is not None else "—"
        display_hr = fmt(data.get("heart_rate"), 0, " BPM") if data.get("heart_rate") is not None else "—"
        display_glucose = (
            fmt(data.get("glucose_level"), 0, " mg/dL")
            if data.get("glucose_level") is not None
            else "—"
        )

        bp_label = classification.get("blood_pressure_label", "—")
        spo2_label = classification.get("spo2_label", "—")
        temp_label = classification.get("temperature_label", "—")
        hr_label = classification.get("heart_rate_label", "—")
        glucose_label = classification.get("glucose_label", "—")
        status_text = str(classification.get("overall_status", "OK")).upper()
        status_label = classification.get("overall_label", "")

    st.markdown(
        f'<p class="status-line"><b style="color:#fff">System Status:</b> '
        f'<span style="color:#fb923c">{html_lib.escape(status_text)}</span> — '
        f"<i>{html_lib.escape(str(status_label))}</i></p>",
        unsafe_allow_html=True,
    )

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        render_html(vital_card("🩺 Blood pressure", display_bp, f"mmHg · {bp_label}", "vc-bp"))
    with r1c2:
        render_html(vital_card("💧 Oxygen (SpO₂)", display_spo2, spo2_label, "vc-spo2"))
    with r1c3:
        render_html(vital_card("🌡️ Temperature", display_temp, temp_label, "vc-temp"))

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        render_html(vital_card("❤️ Heart rate", display_hr, hr_label, "vc-hr"))
    with r2c2:
        render_html(vital_card("🩸 Glucose", display_glucose, f"{glucose_label} · mg/dL", "vc-glucose"))

    st.markdown(
        '<p class="vc-footer">In a real app these values would come from connected devices. '
        "This demo does not measure vital signs.</p>",
        unsafe_allow_html=True,
    )

    tab_alerts, tab_guidance = st.tabs(["🚨 Threshold Alerts", "💡 Clinical Guidance"])

    with tab_alerts:
        if not alerts:
            st.info("No threshold alerts identified.")
        else:
            for alert in alerts:
                severity = alert.get("severity", "low")
                alert_type = str(alert.get("type", "Alert")).replace("_", " ").title()
                message = alert.get("message", "")
                next_action = alert.get("next_action", "")
                css = severity_class(severity)
                render_html(
                    f'<div class="alert-card {css}">'
                    f'<div style="color:#fff;font-weight:700;font-size:0.88rem;margin-bottom:6px">'
                    f"{html_lib.escape(str(severity).upper())} · {html_lib.escape(alert_type)}"
                    f"</div>"
                    f'<div style="color:#d1d5db;font-size:0.84rem;margin-bottom:6px">'
                    f"{html_lib.escape(str(message))}"
                    f"</div>"
                    f'<div style="color:#9ca3af;font-size:0.78rem">'
                    f"<b>Recommended action:</b> {html_lib.escape(str(next_action))}"
                    f"</div>"
                    f"</div>"
                )

    with tab_guidance:
        st.markdown("##### 💡 Clinical recommendations")
        if recommendations:
            for rec in recommendations:
                topic = rec.get("topic") or "Guidance"
                text = rec.get("text") or ""
                st.markdown(f"- **{html_lib.escape(str(topic))}:** {html_lib.escape(str(text))}")
        else:
            st.caption("No specific recommendations at this time.")

        if first_aid.get("has_emergency"):
            render_html(
                '<div class="vc-emergency">'
                '<div class="emergency-title">🚨 Emergency first-aid guidance</div>'
                '<div class="emergency-sub">If in doubt, call your local emergency number now.</div>'
                "</div>"
            )
            for guidance in first_aid.get("guidance", []):
                st.markdown(f"### 🚨 {guidance.get('title', 'First aid')}")
                for step in guidance.get("steps", []):
                    st.markdown(f"- {step}")
            if first_aid.get("disclaimer"):
                st.caption(first_aid["disclaimer"])
        else:
            st.success("No urgent first-aid requirements identified.")

# ------------------------------------------------------------
# Analyze action
# ------------------------------------------------------------
if analyze:
    # Prefer widget values (what user currently sees)
    payload = {
        "patient_id": int(patient_id) if patient_id > 0 else None,
        "age": int(age) if age > 0 else None,
        "heart_rate": float(heart_rate) if heart_rate > 0 else None,
        "glucose_level": float(glucose) if glucose > 0 else None,
        "systolic_bp": int(systolic_bp) if systolic_bp > 0 else None,
        "diastolic_bp": int(diastolic_bp) if diastolic_bp > 0 else None,
        "spo2": float(spo2) if spo2 > 0 else None,
        "temperature_c": float(temperature) if temperature > 0 else None,
        "symptoms": (symptoms or "").strip() or None,
        "known_condition": (known_condition or "").strip() or None,
    }

    with st.spinner("Analyzing vitals…"):
        assessment = run_full_assessment(payload)

    vitals_result = assessment.get("vitals", {})
    if vitals_result.get("status") != "ok":
        st.error(
            vitals_result.get("data", {}).get("message")
            or vitals_result.get("data", {}).get("validation_error")
            or "Invalid input."
        )
    else:
        st.session_state["assessment"] = assessment
        try:
            save_measurement(payload, assessment)
        except Exception as exc:
            st.warning(f"Analysis completed, but history could not be saved: {exc}")
        st.rerun()

# ------------------------------------------------------------
# Synthetic patient directory + history + footer
# ------------------------------------------------------------
st.divider()

with st.expander("👥 Synthetic patient directory (demo data)", expanded=False):
    st.caption(
        "All records are fictional and only loosely realistic for demonstration. "
        "Do not enter real personal health information."
    )
    # Merge patients + vitals for a clear table
    directory = patients.merge(vitals, on="patient_id", how="left")
    show_cols = [
        "patient_id",
        "name",
        "age",
        "condition",
        "heart_rate",
        "glucose_level",
        "blood_pressure",
        "spo2",
        "temperature_c",
    ]
    st.dataframe(
        directory[show_cols].rename(
            columns={
                "patient_id": "ID",
                "name": "Name",
                "age": "Age",
                "condition": "Condition",
                "heart_rate": "HR (bpm)",
                "glucose_level": "Glucose",
                "blood_pressure": "BP",
                "spo2": "SpO₂ %",
                "temperature_c": "Temp °C",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("📋 Measurement History"):
    try:
        history_result = get_measurement_history(limit=50)
        items = history_result.get("data", {}).get("items", [])
        if items:
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
        else:
            st.info("No measurement history yet. Analyze a patient to save a reading.")
    except Exception as exc:
        st.warning(f"History unavailable: {exc}")

st.markdown(
    '<p style="text-align:center;color:#6b7280;font-size:0.75rem;padding:16px 0 4px 0">'
    "Smart Healthcare Assistant · Educational prototype · Synthetic/demo data only<br>"
    "This application does not diagnose, prescribe, or replace a licensed healthcare professional."
    "</p>",
    unsafe_allow_html=True,
)
