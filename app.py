
import streamlit as st
import pandas as pd
import html

from healthcare_core import (
    run_full_assessment,
    get_measurement_history,
    save_measurement,
    get_patient_record,
    monitor_vitals,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VitalCare",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK VITALCARE THEME
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, sans-serif;
}

.stApp {
    background: #0b0b0d;
    color: #ffffff;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}


/* ============================================================
   LEFT INPUT PANEL
   ============================================================ */

.input-panel {
    background: #161619;
    border: 1px solid #222228;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
}

.input-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 18px;
}


/* ============================================================
   VITALCARE HEADER
   ============================================================ */

.vitalcare-header {
    background: #111113;
    border: 1px solid #1f1f23;
    border-radius: 14px;
    padding: 18px 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.vitalcare-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.vitalcare-icon {
    background: #116124;
    color: #e2fceb;
    padding: 10px 12px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 1.1rem;
}

.vitalcare-name {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
}

.vitalcare-subtitle {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 2px;
}

.demo-badge {
    background: #222228;
    color: #fb923c;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid #2f2f37;
}


/* ============================================================
   VITAL CARDS
   ============================================================ */

.vc-card {
    background: #161619;
    border: 1px solid #222228;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    min-height: 145px;
    margin-bottom: 16px;
}

.vc-title {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 10px;
}

.vc-bp {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f87171;
    line-height: 1.1;
}

.vc-spo2 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.1;
}

.vc-temp {
    font-size: 2.2rem;
    font-weight: 700;
    color: #fb923c;
    line-height: 1.1;
}

.vc-hr {
    font-size: 2.2rem;
    font-weight: 700;
    color: #c084fc;
    line-height: 1.1;
}

.vc-subtext {
    color: #9ca3af;
    font-size: 0.78rem;
    margin-top: 6px;
}


/* ============================================================
   FOOTER / DISCLAIMER
   ============================================================ */

.vc-footer {
    color: #6b7280;
    font-size: 0.78rem;
    margin-top: 5px;
    margin-bottom: 20px;
    line-height: 1.4;
}


/* ============================================================
   ALERT CARD
   ============================================================ */

.alert-card {
    background: #161619;
    border: 1px solid #222228;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}

.alert-critical {
    border-left: 5px solid #dc2626;
}

.alert-high {
    border-left: 5px solid #ea580c;
}

.alert-moderate {
    border-left: 5px solid #d97706;
}

.alert-low {
    border-left: 5px solid #2563eb;
}


/* ============================================================
   RECOMMENDATIONS
   ============================================================ */

.vc-rec-card {
    background: #161619;
    border: 1px solid #222228;
    border-radius: 14px;
    padding: 20px;
    margin-top: 12px;
}

.rec-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 12px;
}

.rec-item {
    margin-bottom: 12px;
    color: #d1d5db;
    font-size: 0.85rem;
}

.rec-topic {
    color: #e5e7eb;
    font-weight: 600;
}


/* ============================================================
   EMERGENCY
   ============================================================ */

.vc-emergency-card {
    background: #1c1113;
    border: 1px solid #7f1d1d;
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
    box-shadow: 0 4px 16px rgba(239,68,68,0.15);
}

.emergency-title {
    color: #f87171;
    font-size: 0.95rem;
    font-weight: 700;
}

.emergency-subtitle {
    color: #fca5a5;
    font-size: 0.82rem;
    margin-top: 8px;
}


/* ============================================================
   STREAMLIT INPUTS
   ============================================================ */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #111113 !important;
    color: #ffffff !important;
    border: 1px solid #2f2f37 !important;
    border-radius: 8px !important;
}

label {
    color: #d1d5db !important;
}

.stButton > button {
    background: #166534 !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 48px !important;
}

.stButton > button:hover {
    background: #15803d !important;
}


/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    background: #111113;
    border-radius: 10px;
    padding: 5px;
    gap: 5px;
}

.stTabs [data-baseweb="tab"] {
    color: #9ca3af;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
}


/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #222228;
    border-radius: 10px;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background: #0f0f11;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_value(value, default="—"):
    if value is None or value == "":
        return default
    return value


def format_number(value, decimals=1):
    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}"
    except:
        return str(value)


def vital_card(title, value, subtitle, css_class):
    return f"""
    <div class="vc-card">
        <div class="vc-title">{title}</div>
        <div class="{css_class}">{value}</div>
        <div class="vc-subtext">{subtitle}</div>
    </div>
    """


def severity_class(severity):
    severity = str(severity).lower()

    if severity == "critical":
        return "alert-critical"

    if severity == "high":
        return "alert-high"

    if severity == "moderate":
        return "alert-moderate"

    return "alert-low"


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="vitalcare-header">

    <div class="vitalcare-left">

        <div class="vitalcare-icon">
            🩺
        </div>

        <div>
            <div class="vitalcare-name">
                VitalCare
            </div>

            <div class="vitalcare-subtitle">
                Patient health dashboard
            </div>
        </div>

    </div>

    <div class="demo-badge">
        Demo
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LAYOUT
# LEFT = INPUTS
# RIGHT = DASHBOARD
# ============================================================

left, right = st.columns([1, 2], gap="large")


# ============================================================
# LEFT COLUMN
# ============================================================

with left:

    st.markdown("""
    <div class="input-panel">

        <div class="input-title">
            ⚙️ Patient Vitals Input
        </div>

    </div>
    """, unsafe_allow_html=True)

    patient_id = st.number_input(
        "Patient ID (Optional)",
        min_value=0,
        value=0,
        step=1
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=33,
        step=1
    )

    col1, col2 = st.columns(2)

    with col1:
        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=0,
            max_value=260,
            value=120,
            step=1
        )

    with col2:
        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=0,
            max_value=180,
            value=80,
            step=1
        )

    col1, col2 = st.columns(2)

    with col1:
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=0,
            max_value=250,
            value=72,
            step=1
        )

    with col2:
        spo2 = st.number_input(
            "SpO₂ (%)",
            min_value=0,
            max_value=100,
            value=98,
            step=1
        )

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temp (°C)",
            min_value=0.0,
            max_value=43.0,
            value=36.8,
            step=0.1
        )

    with col2:
        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=700,
            value=95,
            step=1
        )

    symptoms = st.text_area(
        "Symptoms",
        placeholder="Describe symptoms...",
        height=90
    )

    known_condition = st.text_input(
        "Known Condition",
        value=""
    )

    analyze = st.button(
        "🔍 Analyze Vitals",
        use_container_width=True
    )

    st.markdown("""
    <div class="vc-footer">
        Enter the patient's measurements and click
        <b>Analyze Vitals</b> to generate the dashboard.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RIGHT COLUMN
# ============================================================

with right:

    # --------------------------------------------------------
    # Default values before analysis
    # --------------------------------------------------------

    if "assessment" not in st.session_state:

        display_bp = "120/80"
        display_spo2 = "98%"
        display_temp = "36.8°C"
        display_hr = "72 BPM"

        bp_label = "Example"
        spo2_label = "Example reading"
        temp_label = "Example reading"
        hr_label = "Example reading"

        alerts = []
        recommendations = []
        first_aid = {}
        status_text = "READY"
        status_label = "Waiting for analysis"

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

        # care can either be the full object or contain data
        if isinstance(care_result, dict) and "data" in care_result:
            care = care_result.get("data") or {}
        else:
            care = care_result

        recommendations = care.get("suggestions", [])

        first_aid = care.get("first_aid", {})

        display_bp = (
            f"{data.get('systolic_bp')}/{data.get('diastolic_bp')}"
            if data.get("systolic_bp") is not None
            and data.get("diastolic_bp") is not None
            else "—"
        )

        display_spo2 = (
            f"{format_number(data.get('spo2'), 0)}%"
            if data.get("spo2") is not None
            else "—"
        )

        display_temp = (
            f"{format_number(data.get('temperature_c'), 1)}°C"
            if data.get("temperature_c") is not None
            else "—"
        )

        display_hr = (
            f"{format_number(data.get('heart_rate'), 0)} BPM"
            if data.get("heart_rate") is not None
            else "—"
        )

        bp_label = classification.get(
            "blood_pressure_label",
            "—"
        )

        spo2_label = classification.get(
            "spo2_label",
            "—"
        )

        temp_label = classification.get(
            "temperature_label",
            "—"
        )

        hr_label = classification.get(
            "heart_rate_label",
            "—"
        )

        status_text = str(
            classification.get(
                "overall_status",
                "OK"
            )
        ).upper()

        status_label = classification.get(
            "overall_label",
            ""
        )


    # ========================================================
    # STATUS
    # ========================================================

    st.markdown(
        f"""
        <div style="
            color:#9ca3af;
            font-size:0.85rem;
            margin-bottom:14px;
        ">
            <b style="color:#ffffff;">System Status:</b>
            <span style="color:#fb923c;">
                {html.escape(status_text)}
            </span>
            —
            <i>{html.escape(str(status_label))}</i>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # VITAL CARDS — SAME STRUCTURE AS GRADIO
    # ========================================================

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            vital_card(
                "🩺 Blood pressure",
                display_bp,
                f"mmHg · {bp_label}",
                "vc-bp"
            ),
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            vital_card(
                "💧 Oxygen (SpO₂)",
                display_spo2,
                spo2_label,
                "vc-spo2"
            ),
            unsafe_allow_html=True
        )


    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            vital_card(
                "🌡️ Temperature",
                display_temp,
                temp_label,
                "vc-temp"
            ),
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            vital_card(
                "❤️ Heart rate",
                display_hr,
                hr_label,
                "vc-hr"
            ),
            unsafe_allow_html=True
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown("""
    <div class="vc-footer">
        In a real app, the values would come from connected
        measuring devices or a validated measurement method.
        This mock dashboard does not measure vital signs.
    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # TABS
    # ========================================================

    tab_alerts, tab_guidance = st.tabs([
        "🚨 Threshold Alerts",
        "💡 Clinical Guidance"
    ])


    # ========================================================
    # THRESHOLD ALERTS
    # ========================================================

    with tab_alerts:

        if not alerts:

            st.markdown("""
            <div class="vc-rec-card">
                <div style="
                    color:#9ca3af;
                    font-size:0.85rem;
                ">
                    No threshold alerts identified.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            for alert in alerts:

                severity = alert.get(
                    "severity",
                    "low"
                )

                alert_type = str(
                    alert.get("type", "Alert")
                ).replace("_", " ").title()

                message = alert.get(
                    "message",
                    ""
                )

                next_action = alert.get(
                    "next_action",
                    ""
                )

                css = severity_class(severity)

                st.markdown(
                    f"""
                    <div class="alert-card {css}">

                        <div style="
                            color:#ffffff;
                            font-weight:700;
                            font-size:0.88rem;
                            margin-bottom:6px;
                        ">
                            {html.escape(str(severity).upper())}
                            ·
                            {html.escape(alert_type)}
                        </div>

                        <div style="
                            color:#d1d5db;
                            font-size:0.84rem;
                            margin-bottom:6px;
                        ">
                            {html.escape(str(message))}
                        </div>

                        <div style="
                            color:#9ca3af;
                            font-size:0.78rem;
                        ">
                            <b>Recommended action:</b>
                            {html.escape(str(next_action))}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ========================================================
    # CLINICAL GUIDANCE
    # ========================================================

    with tab_guidance:

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        st.markdown("""
        <div class="vc-rec-card">

            <div class="rec-title">
                💡 Clinical Recommendations
            </div>

        """, unsafe_allow_html=True)

        if recommendations:

            for recommendation in recommendations:

                topic = recommendation.get(
                    "topic",
                    "Guidance"
                )

                text = recommendation.get(
                    "text",
                    ""
                )

                st.markdown(
                    f"""
                    <div class="rec-item">
                        • <span class="rec-topic">
                            {html.escape(str(topic))}
                          </span>:
                        {html.escape(str(text))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.markdown("""
                <div style="
                    color:#9ca3af;
                    font-size:0.85rem;
                ">
                    No specific recommendations at this time.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # Emergency First Aid
        # ----------------------------------------------------

        if first_aid.get("has_emergency"):

            st.markdown("""
            <div class="vc-emergency-card">

                <div class="emergency-title">
                    🚨 Emergency first-aid guidance
                </div>

                <div class="emergency-subtitle">
                    If in doubt, call your local emergency number now.
                </div>

            </div>
            """, unsafe_allow_html=True)

            guidance_items = first_aid.get(
                "guidance",
                []
            )

            for guidance in guidance_items:

                title = guidance.get(
                    "title",
                    "First aid"
                )

                st.markdown(
                    f"### 🚨 {title}"
                )

                for step in guidance.get(
                    "steps",
                    []
                ):

                    st.markdown(
                        f"- {step}"
                    )

            if first_aid.get("disclaimer"):

                st.caption(
                    first_aid["disclaimer"]
                )

        else:

            st.markdown("""
            <div class="vc-rec-card">

                <div style="
                    color:#ffffff;
                    font-size:0.95rem;
                    font-weight:600;
                    margin-bottom:6px;
                ">
                    🛡️ First-Aid Status
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:0.85rem;
                ">
                    No urgent first-aid requirements identified.
                </div>

            </div>
            """, unsafe_allow_html=True)


# ============================================================
# ANALYZE BUTTON ACTION
# ============================================================

if analyze:

    payload = {
        "patient_id": int(patient_id)
        if patient_id > 0 else None,

        "age": int(age)
        if age > 0 else None,

        "heart_rate": float(heart_rate)
        if heart_rate > 0 else None,

        "glucose_level": float(glucose)
        if glucose > 0 else None,

        "systolic_bp": int(systolic_bp)
        if systolic_bp > 0 else None,

        "diastolic_bp": int(diastolic_bp)
        if diastolic_bp > 0 else None,

        "spo2": float(spo2)
        if spo2 > 0 else None,

        "temperature_c": float(temperature)
        if temperature > 0 else None,

        "symptoms": symptoms.strip()
        if symptoms.strip() else None,

        "known_condition": known_condition.strip()
        if known_condition.strip() else None,
    }

    with st.spinner("Analyzing vitals..."):

        assessment = run_full_assessment(
            payload
        )

    vitals_result = assessment.get(
        "vitals",
        {}
    )

    if vitals_result.get("status") != "ok":

        st.error(
            vitals_result.get(
                "data",
                {}
            ).get("message")
            or
            vitals_result.get(
                "data",
                {}
            ).get("validation_error")
            or
            "Invalid input."
        )

    else:

        st.session_state["assessment"] = assessment

        try:
            save_measurement(
                payload,
                assessment
            )
        except Exception as e:
            st.warning(
                f"Analysis completed, but measurement history could not be saved: {e}"
            )

        st.rerun()


# ============================================================
# OPTIONAL DATA / HISTORY
# ============================================================

st.divider()

with st.expander("📋 Measurement History"):

    try:

        history_result = get_measurement_history(
            limit=50
        )

        items = (
            history_result
            .get("data", {})
            .get("items", [])
        )

        if items:

            st.dataframe(
                pd.DataFrame(items),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No measurement history available yet."
            )

    except Exception as e:

        st.warning(
            f"Measurement history unavailable: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    text-align:center;
    color:#6b7280;
    font-size:0.75rem;
    padding:20px 0 5px 0;
">

    VitalCare · Educational prototype · Synthetic/demo data only

    <br>

    This application does not diagnose, prescribe, or replace
    a licensed healthcare professional.

</div>
""", unsafe_allow_html=True)