"""
Smart Healthcare Assistant - Core Backend (MCP-ready tool layer)
==================================================================
This module contains ALL business logic: schemas, synthetic data, the five
MCP-style tools, the orchestrator, and the automated tests. Both the
Streamlit app (streamlit_app.py) and the Gradio app (gradio_app.py) import
directly from this file, so the two UIs can never silently drift apart.

Educational prototype only - synthetic data, not medical advice, not a
diagnosis, not a prescription. Always seek local emergency services for a
real emergency.
"""

from __future__ import annotations

import re
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# Optional semantic-search stack (Chroma + sentence-transformers).
# Falls back to a lightweight keyword-overlap search if either package or an
# internet connection (for the embedding model download) isn't available, so
# the notebook/app never hard-crashes just because retrieval is degraded.
# ---------------------------------------------------------------------------
_SEMANTIC_SEARCH_AVAILABLE = True
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised only when deps are missing
    _SEMANTIC_SEARCH_AVAILABLE = False


# ===========================================================================
# 1. Schemas
# ===========================================================================
class PatientIdInput(BaseModel):
    patient_id: int = Field(..., ge=1, description="Synthetic patient identifier")


class UserHealthInput(BaseModel):
    """Validated shape for vitals/symptoms a user types into the form themselves.

    Every field is optional except that at least one signal (a vital or
    symptoms) must be present - that check happens in `analyze_user_vitals`
    so the schema itself stays simple and reusable.
    """
    patient_id: Optional[int] = Field(None, ge=1, description="Optional synthetic patient ID used as a baseline")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    heart_rate: Optional[float] = Field(None, ge=20, le=250, description="Heart rate in bpm")
    glucose_level: Optional[float] = Field(None, ge=20, le=700, description="Blood glucose in mg/dL")
    systolic_bp: Optional[int] = Field(None, ge=60, le=260, description="Systolic blood pressure in mmHg")
    diastolic_bp: Optional[int] = Field(None, ge=30, le=180, description="Diastolic blood pressure in mmHg")
    spo2: Optional[float] = Field(None, ge=50, le=100, description="Oxygen saturation (SpO2) in %")
    temperature_c: Optional[float] = Field(None, ge=30, le=43, description="Body temperature in Celsius")
    symptoms: Optional[str] = Field(None, max_length=500, description="Free-text symptoms described by the user")
    known_condition: Optional[str] = Field(None, max_length=120, description="Self-reported condition, e.g. Diabetes")

    @model_validator(mode="after")
    def _validate_blood_pressure_pair(self):
        if (self.systolic_bp is None) != (self.diastolic_bp is None):
            raise ValueError("Enter both systolic and diastolic blood pressure, or leave both blank.")
        if self.systolic_bp is not None and self.diastolic_bp is not None:
            if self.systolic_bp <= self.diastolic_bp:
                raise ValueError("Systolic blood pressure must be greater than diastolic blood pressure.")
        return self

    @field_validator("symptoms", "known_condition")
    @classmethod
    def _strip_text(cls, v):
        return v.strip() if isinstance(v, str) else v


class Alert(BaseModel):
    type: str
    severity: Literal["low", "moderate", "high", "critical"]
    message: str
    next_action: str


class ToolResult(BaseModel):
    tool_name: str
    request_id: str
    timestamp_utc: str
    status: Literal["ok", "not_found", "error"]
    data: Dict[str, Any] = {}
    alerts: List[Alert] = []
    disclaimer: str = "Educational decision support only. Not medical advice, a diagnosis, or a prescription."


audit_log: List[Dict[str, Any]] = []


def audit(tool_name: str, patient_id: Optional[int], status: str) -> None:
    audit_log.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
        "tool_name": tool_name,
        "patient_id": patient_id,
        "status": status,
        "data_classification": "demo_or_user_entered_health_data",
    })


def result(tool_name: str, status: str, data: Optional[dict] = None, alerts: Optional[list] = None) -> Dict[str, Any]:
    return ToolResult(
        tool_name=tool_name,
        request_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status=status,
        data=data or {},
        alerts=alerts or [],
    ).model_dump()


SEVERITY_RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}


# ===========================================================================
# 2. Synthetic data and privacy
# ===========================================================================
# All records are fictional and only loosely "realistic" (plausible ranges
# for the stated condition/age) for demonstration purposes. Do not enter real
# personal health information into this notebook, a public Gradio link, or
# an unapproved vector database.

patients = pd.DataFrame([
    {"patient_id": 201, "name": "Alice",   "age": 54, "condition": "Type 2 Diabetes"},
    {"patient_id": 202, "name": "Bob",     "age": 61, "condition": "Hypertension"},
    {"patient_id": 203, "name": "Charlie", "age": 29, "condition": "Healthy"},
    {"patient_id": 204, "name": "John",    "age": 67, "condition": "Cardiovascular Disease"},
    {"patient_id": 205, "name": "Maria",   "age": 42, "condition": "Asthma / COPD"},
    {"patient_id": 206, "name": "Fatima",  "age": 35, "condition": "Anemia"},
    {"patient_id": 207, "name": "Omar",    "age": 38, "condition": "Hyperthyroidism"},
    {"patient_id": 208, "name": "Grace",   "age": 55, "condition": "Hypothyroidism"},
    {"patient_id": 209, "name": "Daniel",  "age": 63, "condition": "Chronic Kidney Disease"},
    {"patient_id": 210, "name": "Priya",   "age": 47, "condition": "Obesity / Metabolic Syndrome"},
    {"patient_id": 211, "name": "Noah",    "age": 26, "condition": "Anxiety / Panic Disorder"},
    {"patient_id": 212, "name": "Layla",   "age": 70, "condition": "Suspected Sepsis / Acute Infection"},
    {"patient_id": 213, "name": "Sara",    "age": 29, "condition": "Pregnancy (3rd trimester)"},
    {"patient_id": 214, "name": "Ahmed",   "age": 33, "condition": "Heat Exhaustion / Dehydration"},
])

# heart_rate (bpm), glucose_level (mg/dL), blood_pressure "sys/dia" (mmHg),
# spo2 (%), temperature_c (Celsius) - plausible for the stated condition/age.
vitals = pd.DataFrame([
    {"patient_id": 201, "heart_rate": 88,  "glucose_level": 185, "blood_pressure": "138/86", "spo2": 97, "temperature_c": 36.9, "measured_at": "2026-09-04T08:00:00Z"},
    {"patient_id": 202, "heart_rate": 78,  "glucose_level": 105, "blood_pressure": "162/98", "spo2": 96, "temperature_c": 36.7, "measured_at": "2026-09-04T08:05:00Z"},
    {"patient_id": 203, "heart_rate": 72,  "glucose_level": 92,  "blood_pressure": "118/76", "spo2": 98, "temperature_c": 36.8, "measured_at": "2026-09-04T08:10:00Z"},
    {"patient_id": 204, "heart_rate": 95,  "glucose_level": 140, "blood_pressure": "150/92", "spo2": 94, "temperature_c": 37.0, "measured_at": "2026-09-04T08:15:00Z"},
    {"patient_id": 205, "heart_rate": 102, "glucose_level": 98,  "blood_pressure": "128/82", "spo2": 91, "temperature_c": 37.1, "measured_at": "2026-09-04T08:20:00Z"},
    {"patient_id": 206, "heart_rate": 108, "glucose_level": 88,  "blood_pressure": "105/68", "spo2": 96, "temperature_c": 36.6, "measured_at": "2026-09-04T08:25:00Z"},
    {"patient_id": 207, "heart_rate": 118, "glucose_level": 90,  "blood_pressure": "132/78", "spo2": 98, "temperature_c": 37.4, "measured_at": "2026-09-04T08:30:00Z"},
    {"patient_id": 208, "heart_rate": 58,  "glucose_level": 96,  "blood_pressure": "110/70", "spo2": 97, "temperature_c": 36.2, "measured_at": "2026-09-04T08:35:00Z"},
    {"patient_id": 209, "heart_rate": 82,  "glucose_level": 130, "blood_pressure": "158/94", "spo2": 95, "temperature_c": 36.9, "measured_at": "2026-09-04T08:40:00Z"},
    {"patient_id": 210, "heart_rate": 90,  "glucose_level": 118, "blood_pressure": "142/88", "spo2": 95, "temperature_c": 36.8, "measured_at": "2026-09-04T08:45:00Z"},
    {"patient_id": 211, "heart_rate": 112, "glucose_level": 95,  "blood_pressure": "128/80", "spo2": 98, "temperature_c": 36.9, "measured_at": "2026-09-04T08:50:00Z"},
    {"patient_id": 212, "heart_rate": 122, "glucose_level": 110, "blood_pressure": "88/58",  "spo2": 90, "temperature_c": 39.4, "measured_at": "2026-09-04T08:55:00Z"},
    {"patient_id": 213, "heart_rate": 92,  "glucose_level": 100, "blood_pressure": "124/80", "spo2": 98, "temperature_c": 36.9, "measured_at": "2026-09-04T09:00:00Z"},
    {"patient_id": 214, "heart_rate": 115, "glucose_level": 92,  "blood_pressure": "100/64", "spo2": 96, "temperature_c": 38.6, "measured_at": "2026-09-04T09:05:00Z"},
])


def parse_bp(value: str) -> Optional[tuple]:
    match = re.fullmatch(r"\s*(\d{2,3})/(\d{2,3})\s*", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


# ===========================================================================
# 3. Tool 1: Patient lookup
# ===========================================================================
def get_patient_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        patient_id = PatientIdInput(**payload).patient_id
    except ValidationError as e:
        audit("get_patient_record", None, "error")
        return result("get_patient_record", "error", {"validation_error": str(e)})

    record = patients.loc[patients.patient_id == patient_id]
    if record.empty:
        audit("get_patient_record", patient_id, "not_found")
        return result("get_patient_record", "not_found", {"message": "No synthetic patient record found."})

    audit("get_patient_record", patient_id, "ok")
    return result("get_patient_record", "ok", {"patient": record.iloc[0].to_dict()})


# ===========================================================================
# 4. Shared vitals-evaluation logic (Tool 2 + Tool 2b both call this)
# ===========================================================================
def evaluate_vitals(
    hr: Optional[float] = None,
    glucose: Optional[float] = None,
    systolic: Optional[int] = None,
    diastolic: Optional[int] = None,
    spo2: Optional[float] = None,
    temperature_c: Optional[float] = None,
) -> List[Alert]:
    """Shared threshold logic used by both the synthetic-patient tool and the
    user-entered-vitals tool, so the two paths can never silently drift apart.

    Thresholds are simplified, textbook-style demonstration values - not a
    clinically validated triage tool.
    """
    alerts: List[Alert] = []

    # --- Heart rate ---------------------------------------------------
    if hr is not None:
        if hr >= 130 or hr <= 40:
            alerts.append(Alert(type="heart_rate", severity="critical",
                                 message=f"Heart rate observation: {hr:.0f} bpm is far outside the typical resting range.",
                                 next_action="Seek urgent clinician assessment; use local emergency services if severe symptoms (chest pain, fainting, breathlessness) are present."))
        elif hr > 100:
            alerts.append(Alert(type="heart_rate", severity="moderate",
                                 message=f"Elevated heart rate observation: {hr:.0f} bpm.",
                                 next_action="Recheck measurement after resting a few minutes and discuss with a clinician, especially if symptoms occur."))
        elif hr < 60:
            alerts.append(Alert(type="heart_rate", severity="moderate",
                                 message=f"Low heart rate observation: {hr:.0f} bpm.",
                                 next_action="Recheck measurement; discuss with a clinician if dizziness, fatigue, or fainting occur."))

    # --- Glucose --------------------------------------------------------
    if glucose is not None:
        if glucose >= 400:
            alerts.append(Alert(type="glucose", severity="critical",
                                 message=f"Very high glucose observation: {glucose:.0f} mg/dL.",
                                 next_action="This range can indicate a diabetic emergency (e.g. DKA/HHS). Seek urgent/emergency care promptly, especially with vomiting, confusion, or rapid breathing."))
        elif glucose >= 250:
            alerts.append(Alert(type="glucose", severity="high",
                                 message=f"High glucose observation: {glucose:.0f} mg/dL.",
                                 next_action="Follow the patient-specific care plan and contact a clinician promptly."))
        elif glucose > 140:
            alerts.append(Alert(type="glucose", severity="moderate",
                                 message=f"Elevated glucose observation: {glucose:.0f} mg/dL.",
                                 next_action="Interpret with timing of meals and patient care plan; discuss with a clinician."))
        elif glucose < 54:
            alerts.append(Alert(type="glucose", severity="critical",
                                 message=f"Very low glucose observation: {glucose:.0f} mg/dL.",
                                 next_action="This range can indicate severe hypoglycemia. Follow your emergency low-glucose plan or seek urgent care."))
        elif glucose < 70:
            alerts.append(Alert(type="glucose", severity="moderate",
                                 message=f"Low glucose observation: {glucose:.0f} mg/dL.",
                                 next_action="Recheck and consider a fast-acting sugar snack per your care plan; discuss recurring low readings with a clinician."))

    # --- Blood pressure ---------------------------------------------------
    if systolic is not None and diastolic is not None:
        if systolic >= 180 or diastolic >= 120:
            alerts.append(Alert(type="blood_pressure", severity="critical",
                                 message=f"Very high blood-pressure observation: {systolic}/{diastolic} mmHg (hypertensive crisis range).",
                                 next_action="Seek urgent clinical assessment; use emergency services if chest pain, vision changes, confusion, or severe headache are present."))
        elif systolic <= 80 or diastolic <= 50:
            alerts.append(Alert(type="blood_pressure", severity="critical",
                                 message=f"Very low blood-pressure observation: {systolic}/{diastolic} mmHg.",
                                 next_action="This range can indicate shock, especially with dizziness, confusion, or fainting. Seek emergency care."))
        elif systolic >= 140 or diastolic >= 90:
            alerts.append(Alert(type="blood_pressure", severity="moderate",
                                 message=f"Elevated blood-pressure observation: {systolic}/{diastolic} mmHg.",
                                 next_action="Re-measure correctly (seated, rested) and review a pattern of readings with a clinician."))
        elif systolic < 90 or diastolic < 60:
            alerts.append(Alert(type="blood_pressure", severity="moderate",
                                 message=f"Low blood-pressure observation: {systolic}/{diastolic} mmHg.",
                                 next_action="Recheck measurement; discuss with a clinician if dizziness, lightheadedness, or fainting occur."))

    # --- Oxygen saturation (SpO2) ---------------------------------------
    if spo2 is not None:
        if spo2 < 90:
            alerts.append(Alert(type="spo2", severity="critical",
                                 message=f"Low oxygen-saturation observation: {spo2:.0f}%.",
                                 next_action="This range can indicate significant hypoxemia. Seek emergency care immediately."))
        elif spo2 < 94:
            alerts.append(Alert(type="spo2", severity="high",
                                 message=f"Reduced oxygen-saturation observation: {spo2:.0f}%.",
                                 next_action="Recheck measurement and seek prompt clinical assessment, especially with breathlessness."))
        elif spo2 < 96:
            alerts.append(Alert(type="spo2", severity="moderate",
                                 message=f"Mildly reduced oxygen-saturation observation: {spo2:.0f}%.",
                                 next_action="Recheck measurement; mention it to a clinician if it persists or you feel breathless."))

    # --- Temperature ------------------------------------------------------
    if temperature_c is not None:
        if temperature_c >= 40.0:
            alerts.append(Alert(type="temperature", severity="critical",
                                 message=f"Very high temperature observation: {temperature_c:.1f}\u00b0C.",
                                 next_action="This range can indicate heat stroke or a severe infection. Seek emergency care immediately."))
        elif temperature_c < 35.0:
            alerts.append(Alert(type="temperature", severity="critical",
                                 message=f"Very low temperature observation: {temperature_c:.1f}\u00b0C.",
                                 next_action="This range can indicate hypothermia. Seek emergency care immediately."))
        elif temperature_c >= 38.0:
            alerts.append(Alert(type="temperature", severity="moderate",
                                 message=f"Fever observation: {temperature_c:.1f}\u00b0C.",
                                 next_action="Rest, fluids, and monitor; discuss with a clinician if it persists beyond a couple of days or is very high."))
        elif temperature_c < 36.1:
            alerts.append(Alert(type="temperature", severity="moderate",
                                 message=f"Mildly low temperature observation: {temperature_c:.1f}\u00b0C.",
                                 next_action="Warm up, recheck measurement, and mention it to a clinician if it recurs."))

    return alerts


def classify_value(value: Optional[float], low_crit, low_mod, high_mod, high_crit) -> str:
    """Small helper to label a single vital as Critical/Low/Normal/Elevated/High
    for the UI. Uses the same cut points as evaluate_vitals so labels and
    alerts always agree."""
    if value is None:
        return "Not entered"
    if (low_crit is not None and value <= low_crit) or (high_crit is not None and value >= high_crit):
        return "Critical"
    if low_mod is not None and value < low_mod:
        return "Low"
    if high_mod is not None and value > high_mod:
        return "Elevated"
    return "Normal"


def classify_vitals(hr, glucose, systolic, diastolic, spo2, temperature_c, alerts: List[Alert]) -> Dict[str, str]:
    labels = {
        "heart_rate_label": classify_value(hr, 40, 60, 100, 130),
        "glucose_label": classify_value(glucose, 54, 70, 140, 400) if glucose is not None else "Not entered",
        "blood_pressure_label": (
            "Not entered" if systolic is None or diastolic is None else
            "Critical" if (systolic >= 180 or diastolic >= 120 or systolic <= 80 or diastolic <= 50) else
            "Low" if (systolic < 90 or diastolic < 60) else
            "Elevated" if (systolic >= 140 or diastolic >= 90) else
            "Normal"
        ),
        "spo2_label": (
            "Not entered" if spo2 is None else
            "Critical" if spo2 < 90 else
            "Low" if spo2 < 94 else
            "Slightly low" if spo2 < 96 else
            "Normal"
        ),
        "temperature_label": classify_value(temperature_c, 35.0, 36.1, 38.0, 40.0) if temperature_c is not None else "Not entered",
    }

    if not alerts:
        entered_any = any(v is not None for v in [hr, glucose, systolic, diastolic, spo2, temperature_c])
        labels["overall_label"] = (
            "No configured threshold alerts were triggered by the entered readings. This does not rule out illness or replace clinical assessment."
            if entered_any else "No vitals entered."
        )
        labels["overall_status"] = "healthy" if entered_any else "no_data"
    else:
        top_severity = max(alerts, key=lambda a: SEVERITY_RANK[a.severity]).severity
        if top_severity == "critical":
            labels["overall_label"] = "Possible medical emergency detected in the entered readings."
            labels["overall_status"] = "emergency"
        elif top_severity == "high":
            labels["overall_label"] = "Readings suggest an urgent issue that needs prompt clinical attention."
            labels["overall_status"] = "urgent"
        else:
            labels["overall_label"] = "Some readings are outside the typical range - worth monitoring and discussing with a clinician."
            labels["overall_status"] = "monitor"
    return labels


# ===========================================================================
# 5. Tool 2: Vital-sign monitoring (synthetic patient)
# ===========================================================================
def monitor_vitals(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        patient_id = PatientIdInput(**payload).patient_id
    except ValidationError as e:
        audit("monitor_vitals", None, "error")
        return result("monitor_vitals", "error", {"validation_error": str(e)})

    record = vitals.loc[vitals.patient_id == patient_id]
    if record.empty:
        audit("monitor_vitals", patient_id, "not_found")
        return result("monitor_vitals", "not_found", {"message": "No synthetic vitals found for this patient."})

    row = record.iloc[0].to_dict()
    bp = parse_bp(row["blood_pressure"])
    systolic, diastolic = bp if bp else (None, None)

    alerts = evaluate_vitals(
        hr=row.get("heart_rate"), glucose=row.get("glucose_level"),
        systolic=systolic, diastolic=diastolic,
        spo2=row.get("spo2"), temperature_c=row.get("temperature_c"),
    )
    audit("monitor_vitals", patient_id, "ok")
    return result("monitor_vitals", "ok", {"vitals": row}, alerts)


# ===========================================================================
# 6. Tool 2b: User-entered vitals analysis
# ===========================================================================
SYMPTOM_EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "severe shortness of breath",
    "fainted", "faint", "slurred speech", "one side of my body", "stroke",
    "unconscious", "blue lips", "coughing blood", "seizure", "severe bleeding",
    "anaphylaxis", "throat closing", "can't wake", "cannot wake",
]


def analyze_user_vitals(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = UserHealthInput(**payload)
    except ValidationError as e:
        audit("analyze_user_vitals", None, "error")
        return result("analyze_user_vitals", "error", {"validation_error": str(e)})

    baseline_patient = None
    baseline_vitals = None
    if parsed.patient_id is not None:
        patient_lookup = get_patient_record({"patient_id": parsed.patient_id})
        if patient_lookup["status"] == "ok":
            baseline_patient = patient_lookup["data"]["patient"]
        vitals_lookup = monitor_vitals({"patient_id": parsed.patient_id})
        if vitals_lookup["status"] == "ok":
            baseline_vitals = vitals_lookup["data"]["vitals"]

    hr = parsed.heart_rate if parsed.heart_rate is not None else (baseline_vitals["heart_rate"] if baseline_vitals else None)
    glucose = parsed.glucose_level if parsed.glucose_level is not None else (baseline_vitals["glucose_level"] if baseline_vitals else None)
    spo2 = parsed.spo2 if parsed.spo2 is not None else (baseline_vitals["spo2"] if baseline_vitals else None)
    temperature_c = parsed.temperature_c if parsed.temperature_c is not None else (baseline_vitals["temperature_c"] if baseline_vitals else None)

    if parsed.systolic_bp is not None and parsed.diastolic_bp is not None:
        systolic, diastolic = parsed.systolic_bp, parsed.diastolic_bp
    elif baseline_vitals is not None:
        parsed_bp = parse_bp(baseline_vitals["blood_pressure"])
        systolic, diastolic = parsed_bp if parsed_bp else (None, None)
    else:
        systolic, diastolic = None, None

    has_any_vital = any(v is not None for v in [hr, glucose, systolic, diastolic, spo2, temperature_c])
    if not has_any_vital and not (parsed.symptoms or "").strip():
        audit("analyze_user_vitals", parsed.patient_id, "error")
        return result("analyze_user_vitals", "error",
                       {"message": "Enter at least one vital sign or a symptom description to analyze."})

    alerts = evaluate_vitals(hr, glucose, systolic, diastolic, spo2, temperature_c)

    symptom_text = (parsed.symptoms or "").lower()
    if any(keyword in symptom_text for keyword in SYMPTOM_EMERGENCY_KEYWORDS):
        alerts.insert(0, Alert(
            type="symptom_flag",
            severity="critical",
            message="The described symptoms may indicate a medical emergency.",
            next_action="If this is a real emergency, contact local emergency services immediately. This tool cannot assess emergencies."
        ))

    condition = parsed.known_condition or (baseline_patient["condition"] if baseline_patient else None)
    classification = classify_vitals(hr, glucose, systolic, diastolic, spo2, temperature_c, alerts)

    data = {
        "patient_id": parsed.patient_id,
        "age": parsed.age if parsed.age is not None else (baseline_patient["age"] if baseline_patient else None),
        "condition": condition,
        "heart_rate": hr,
        "glucose_level": glucose,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "spo2": spo2,
        "temperature_c": temperature_c,
        "symptoms": parsed.symptoms,
        "used_baseline_patient": baseline_patient is not None,
        "baseline_patient": baseline_patient,
        "classification": classification,
    }

    audit("analyze_user_vitals", parsed.patient_id, "ok")
    return result("analyze_user_vitals", "ok", data, alerts)


# ===========================================================================
# 7. Tool 3: Knowledge retrieval (Chroma, with a graceful offline fallback)
# ===========================================================================
knowledge_base = [
    {"id": "kb1", "text": "Diabetes self-management commonly includes individualized nutrition, activity, glucose monitoring, and a clinician-approved care plan.", "topic": "Type 2 Diabetes", "kind": "lifestyle"},
    {"id": "kb2", "text": "For diabetes, care teams may discuss structured diabetes education, glucose-lowering medication classes such as metformin, and continuous glucose monitoring, individualized to the patient.", "topic": "Type 2 Diabetes", "kind": "medication_class"},
    {"id": "kb3", "text": "Repeated elevated blood-pressure readings should be measured correctly (seated, rested, correct cuff size) and reviewed by a qualified clinician.", "topic": "Hypertension", "kind": "lifestyle"},
    {"id": "kb4", "text": "For hypertension, commonly discussed medication classes include ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics, with the specific choice and dose set by a clinician.", "topic": "Hypertension", "kind": "medication_class"},
    {"id": "kb5", "text": "For cardiovascular disease, care commonly involves cardiac rehabilitation, statin therapy, antiplatelet medication, and structured risk-factor management under cardiology guidance.", "topic": "Cardiovascular Disease", "kind": "medication_class"},
    {"id": "kb6", "text": "A high heart-rate measurement can have many causes; context, symptoms, repeat measurement, and clinician assessment matter before assuming a cause.", "topic": "Cardiovascular Disease", "kind": "lifestyle"},
    {"id": "kb7", "text": "Asthma and COPD education often covers inhaler technique, trigger avoidance, a written action plan, and recognizing when symptoms need urgent review.", "topic": "Asthma / COPD", "kind": "lifestyle"},
    {"id": "kb8", "text": "For asthma or COPD, clinicians commonly discuss bronchodilator and inhaled-corticosteroid medication classes as part of a personalized action plan.", "topic": "Asthma / COPD", "kind": "medication_class"},
    {"id": "kb9", "text": "Anemia education often covers iron-rich or clinician-recommended dietary sources, fatigue management, and follow-up blood testing to track improvement.", "topic": "Anemia", "kind": "lifestyle"},
    {"id": "kb10", "text": "For anemia, care teams may discuss iron supplementation or other targeted therapy once the underlying cause has been identified by a clinician.", "topic": "Anemia", "kind": "medication_class"},
    {"id": "kb11", "text": "Hyperthyroidism education commonly covers monitoring heart rate and weight, avoiding excess iodine, and regular thyroid-function testing.", "topic": "Hyperthyroidism", "kind": "lifestyle"},
    {"id": "kb12", "text": "For hyperthyroidism, clinicians may discuss antithyroid medication, beta-blockers for symptom control, or other therapy chosen after specialist evaluation.", "topic": "Hyperthyroidism", "kind": "medication_class"},
    {"id": "kb13", "text": "Hypothyroidism education commonly covers energy pacing, cold sensitivity, and the importance of consistent, clinician-guided follow-up testing.", "topic": "Hypothyroidism", "kind": "lifestyle"},
    {"id": "kb14", "text": "For hypothyroidism, clinicians commonly discuss thyroid-hormone replacement therapy, individually dosed and monitored with blood tests.", "topic": "Hypothyroidism", "kind": "medication_class"},
    {"id": "kb15", "text": "Chronic kidney disease education often covers fluid and sodium balance, blood-pressure control, and regular monitoring of kidney function.", "topic": "Chronic Kidney Disease", "kind": "lifestyle"},
    {"id": "kb16", "text": "For chronic kidney disease, care commonly involves blood-pressure medication classes and dietary guidance individualized by a nephrology-informed care team.", "topic": "Chronic Kidney Disease", "kind": "medication_class"},
    {"id": "kb17", "text": "Obesity and metabolic-syndrome education often covers gradual activity increases, balanced nutrition patterns, sleep, and behavioral support.", "topic": "Obesity / Metabolic Syndrome", "kind": "lifestyle"},
    {"id": "kb18", "text": "For metabolic syndrome, care teams may discuss weight-management or glucose/lipid-related medication classes when lifestyle measures alone are not enough.", "topic": "Obesity / Metabolic Syndrome", "kind": "medication_class"},
    {"id": "kb19", "text": "Anxiety and panic-related symptoms can include a racing heart and shortness of breath; slow breathing techniques and grounding exercises are commonly taught coping tools.", "topic": "Anxiety / Panic Disorder", "kind": "therapy"},
    {"id": "kb20", "text": "Talk-therapy approaches such as cognitive behavioral therapy can help with stress, anxiety, or coping related to a chronic illness, alongside medical treatment.", "topic": "Anxiety / Panic Disorder", "kind": "therapy"},
    {"id": "kb21", "text": "A fever combined with a fast heart rate and low blood pressure can be an early warning pattern for a serious infection and deserves prompt clinical evaluation.", "topic": "Suspected Sepsis / Acute Infection", "kind": "lifestyle"},
    {"id": "kb22", "text": "Pregnancy monitoring commonly covers blood pressure, weight trends, fetal movement awareness, and routine prenatal visits with an obstetric care team.", "topic": "Pregnancy (3rd trimester)", "kind": "lifestyle"},
    {"id": "kb23", "text": "Heat exhaustion and dehydration education covers moving to a cool place, fluid replacement, rest, and escalating to emergency care if confusion or very high temperature develop.", "topic": "Heat Exhaustion / Dehydration", "kind": "lifestyle"},
    {"id": "kb24", "text": "General wellness checkups usually cover sleep quality, movement, nutrition, and mental-health screening as a starting point before any condition-specific plan.", "topic": "Healthy", "kind": "lifestyle"},
    {"id": "kb25", "text": "General lifestyle measures often discussed for cardiometabolic health include balanced nutrition, regular physical activity, adequate sleep, stress management, and avoiding tobacco.", "topic": "general wellness", "kind": "lifestyle"},
    # --- Emergency / first-aid education (general public-safety information) ---
    {"id": "kb-ea1", "text": "People with severe symptoms such as chest pain, severe shortness of breath, fainting, or stroke-like symptoms should seek emergency help immediately rather than waiting to see if symptoms pass.", "topic": "emergency", "kind": "first_aid"},
    {"id": "kb-ea2", "text": "The FAST check (Face drooping, Arm weakness, Speech difficulty, Time to call emergency services) is a well-known public tool for recognizing possible stroke.", "topic": "emergency", "kind": "first_aid"},
    {"id": "kb-ea3", "text": "For suspected severe hypoglycemia in someone who is awake and able to swallow safely, general public first-aid guidance is fast-acting sugar (juice or glucose tablets) followed by rechecking; if the person is unconscious, nothing should be given by mouth and emergency services should be called.", "topic": "emergency", "kind": "first_aid"},
    {"id": "kb-ea4", "text": "General heat-illness first aid includes moving the person to a cooler place, removing excess clothing, and cooling the skin with water or damp cloths while waiting for emergency help if confusion or very high temperature are present.", "topic": "emergency", "kind": "first_aid"},
    {"id": "kb-ea5", "text": "General hypothermia first aid includes moving the person somewhere warm and dry, removing wet clothing, and warming them gradually with blankets while avoiding rough handling.", "topic": "emergency", "kind": "first_aid"},
]

_search_cache: Dict[str, Any] = {}


def _init_semantic_search():
    if "collection" in _search_cache:
        return _search_cache["collection"], _search_cache["model"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection("healthcare_kb")
    if collection.count() == 0:
        embeddings = model.encode([d["text"] for d in knowledge_base]).tolist()
        collection.add(
            ids=[d["id"] for d in knowledge_base],
            documents=[d["text"] for d in knowledge_base],
            embeddings=embeddings,
            metadatas=[{"topic": d["topic"], "kind": d["kind"]} for d in knowledge_base],
        )
    _search_cache["collection"] = collection
    _search_cache["model"] = model
    return collection, model


def _keyword_search(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Lightweight fallback search used when Chroma / sentence-transformers /
    an internet connection for the embedding model isn't available."""
    q_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for doc in knowledge_base:
        d_words = set(re.findall(r"[a-z0-9]+", (doc["text"] + " " + doc["topic"]).lower()))
        overlap = len(q_words & d_words)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        scored = [(0, doc) for doc in knowledge_base[:n_results]]
    top = scored[:n_results]
    max_score = max((s for s, _ in top), default=1) or 1
    return [
        {"id": doc["id"], "text": doc["text"], "source": "Educational demo knowledge base (keyword fallback)",
         "relevance": round(min(score / max_score, 1.0), 2)}
        for score, doc in top
    ]


def search_knowledge(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = (payload.get("query") or "").strip()
    n_results = int(payload.get("n_results", 3))
    if not query:
        audit("search_knowledge", None, "error")
        return result("search_knowledge", "error", {"message": "A query string is required."})

    try:
        if _SEMANTIC_SEARCH_AVAILABLE:
            collection, model = _init_semantic_search()
            query_embedding = model.encode([query]).tolist()
            res = collection.query(query_embeddings=query_embedding, n_results=n_results)
            matches = []
            for doc_text, doc_id, dist in zip(res["documents"][0], res["ids"][0], res["distances"][0]):
                matches.append({
                    "id": doc_id, "text": doc_text,
                    "source": "Educational demo knowledge base (semantic search)",
                    "relevance": round(max(0.0, 1 - dist), 2),
                })
        else:
            matches = _keyword_search(query, n_results)
    except Exception:
        # Any runtime hiccup with the embedding/vector stack (e.g. no internet
        # to download the model) falls back to keyword search instead of
        # crashing the whole app.
        matches = _keyword_search(query, n_results)

    audit("search_knowledge", None, "ok")
    return result("search_knowledge", "ok", {"query": query, "matches": matches})


# ===========================================================================
# 8. Tool 4: Emergency first-aid guidance (educational, never a prescription)
# ===========================================================================
# General public-safety first-aid information, keyed by the alert `type` +
# `severity` produced by evaluate_vitals / the symptom keyword flag. This is
# textbook / public-health-course-style information (comparable to a Red
# Cross first-aid leaflet) - it never names a personalized drug, dose, or
# brand, and it always tells the person to call local emergency services.
EMERGENCY_FIRST_AID: Dict[str, Dict[str, Any]] = {
    "symptom_flag": {
        "title": "Possible medical emergency (symptom-based)",
        "steps": [
            "Call your local emergency number right away - do not wait to see if it passes.",
            "Keep the person calm, seated or lying down, and stay with them until help arrives.",
            "If a stroke is possible, use the FAST check: Face drooping, Arm weakness, Speech difficulty, Time to call for help.",
            "If they become unresponsive and are not breathing normally, begin CPR if you are trained, and use an AED if one is available.",
            "Do not give food, drink, or any medication unless directed by the emergency dispatcher.",
            "Note when symptoms started so you can tell responders.",
        ],
    },
    "heart_rate": {
        "title": "Very fast or very slow heart rate",
        "steps": [
            "Call your local emergency number if this is accompanied by chest pain, fainting, or severe breathlessness.",
            "Have the person sit or lie down, stay calm, and loosen tight clothing.",
            "Do not give caffeine, stimulants, or any medication not already prescribed to them for this purpose.",
            "If they lose consciousness and stop breathing normally, begin CPR if trained and use an AED if available.",
        ],
    },
    "blood_pressure_high": {
        "title": "Very high blood pressure (hypertensive crisis range)",
        "steps": [
            "Call your local emergency number if there is chest pain, vision changes, confusion, severe headache, or slurred speech.",
            "Have the person sit calmly and avoid any strenuous activity.",
            "Do not give them blood-pressure medication that is not their own prescribed medication.",
        ],
    },
    "blood_pressure_low": {
        "title": "Very low blood pressure / possible shock",
        "steps": [
            "Call your local emergency number, especially with confusion, fainting, or a rapid weak pulse.",
            "Lay the person down and, if there is no suspected injury, raise their legs slightly.",
            "Keep them warm and do not give food or drink if they are drowsy or confused.",
        ],
    },
    "glucose_low": {
        "title": "Very low blood glucose (possible severe hypoglycemia)",
        "steps": [
            "If the person is awake and can safely swallow, give a fast-acting sugar source (juice, regular soda, or glucose tablets).",
            "Recheck how they feel / their glucose after about 15 minutes if a meter is available.",
            "If they are confused, unconscious, or cannot safely swallow, give nothing by mouth and call emergency services immediately.",
            "If they have an emergency glucagon kit and you are trained to use it, follow their personal emergency plan.",
        ],
    },
    "glucose_high": {
        "title": "Very high blood glucose (possible diabetic emergency)",
        "steps": [
            "Call a clinician promptly, or emergency services if there is vomiting, confusion, fruity-smelling breath, or rapid breathing.",
            "Encourage sipping water if the person is fully alert and able to tolerate fluids.",
            "Avoid strenuous exercise until this has been checked.",
        ],
    },
    "spo2": {
        "title": "Low oxygen saturation",
        "steps": [
            "Call your local emergency number, especially with severe breathlessness, confusion, or blue-tinged lips.",
            "Help the person sit upright and loosen tight clothing around the neck and chest.",
            "If they have prescribed supplemental oxygen or a rescue inhaler, help them use it as directed by their own plan.",
            "Do not leave them alone while waiting for help.",
        ],
    },
    "temperature_high": {
        "title": "Very high body temperature (possible heat stroke)",
        "steps": [
            "Call your local emergency number - heat stroke can be life-threatening.",
            "Move the person to a cooler place and remove excess clothing.",
            "Cool their skin with cool water, damp cloths, or fanning, focusing on the neck, armpits, and groin.",
            "Do not give fluids by mouth if they are confused or losing consciousness.",
        ],
    },
    "temperature_low": {
        "title": "Very low body temperature (possible hypothermia)",
        "steps": [
            "Call your local emergency number.",
            "Move the person somewhere warm and dry and remove any wet clothing.",
            "Warm them gradually with blankets; only give warm (not hot) drinks if they are fully alert.",
            "Handle them gently - avoid vigorous rubbing or sudden movement.",
        ],
    },
}


def get_emergency_first_aid(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Given a list of alert dicts (as produced by evaluate_vitals / analyze_user_vitals),
    return the relevant general first-aid guidance for any high/critical alert."""
    guidance = []
    seen_titles = set()
    for a in alerts:
        if a.get("severity") not in ("high", "critical"):
            continue
        atype = a.get("type")
        key = atype
        if atype == "blood_pressure":
            # evaluate_vitals always writes "low blood-pressure" vs "...blood-pressure observation" (high)
            key = "blood_pressure_low" if "low blood-pressure" in a.get("message", "").lower() else "blood_pressure_high"
        elif atype == "glucose":
            key = "glucose_low" if "low" in a.get("message", "").lower() else "glucose_high"
        elif atype == "temperature":
            key = "temperature_low" if "low" in a.get("message", "").lower() else "temperature_high"

        entry = EMERGENCY_FIRST_AID.get(key)
        if entry and entry["title"] not in seen_titles:
            seen_titles.add(entry["title"])
            guidance.append(entry)

    return {
        "has_emergency": len(guidance) > 0,
        "guidance": guidance,
        "disclaimer": (
            "General public first-aid information for education only - not a personalized treatment "
            "plan or a substitute for emergency medical services, CPR training, or a licensed clinician. "
            "If you believe this is a real emergency, contact your local emergency number now."
        ),
    }


# ===========================================================================
# 9. Tool 5: Care recommendations (educational only, not a prescription)
# ===========================================================================
def recommend_care(payload: Dict[str, Any]) -> Dict[str, Any]:
    condition = (payload.get("condition") or "general wellness").strip() or "general wellness"
    alerts = payload.get("alerts") or []
    severities = {a.get("severity") for a in alerts}
    escalation = bool(severities & {"high", "critical"})

    queries = [
        f"Lifestyle guidance for {condition}",
        f"Medication classes commonly discussed for {condition}",
        f"Therapy and coping support for {condition}",
    ]

    seen = set()
    suggestions = []
    for q in queries:
        response = search_knowledge({"query": q, "n_results": 1})
        for m in response.get("data", {}).get("matches", []):
            if m["text"] not in seen:
                seen.add(m["text"])
                suggestions.append(m)

    first_aid = get_emergency_first_aid(alerts) if escalation else {"has_emergency": False, "guidance": [], "disclaimer": ""}

    audit("recommend_care", payload.get("patient_id"), "ok")
    return result(
        "recommend_care",
        "ok",
        {
            "condition": condition,
            "escalation_recommended": escalation,
            "suggestions": suggestions,
            "first_aid": first_aid,
            "disclaimer": (
                "General educational information only, not a prescription or personalized medical "
                "advice. Medication choice, therapy plans, and dosing must come from a licensed "
                "clinician who knows your full history."
            ),
        },
    )


# ===========================================================================
# 10. Orchestrators
# ===========================================================================
def run_healthcare_assistant(patient_id: int) -> Dict[str, Any]:
    """Original preset-patient flow: lookup + vitals + knowledge retrieval."""
    patient = get_patient_record({"patient_id": patient_id})
    vitals_result = monitor_vitals({"patient_id": patient_id})
    if patient["status"] != "ok":
        return {"patient": patient, "vitals": vitals_result, "knowledge": None}

    condition = patient["data"]["patient"]["condition"]
    query = f"Educational guidance for {condition} and abnormal vital signs"
    knowledge = search_knowledge({"query": query})
    return {"patient": patient, "vitals": vitals_result, "knowledge": knowledge}


def run_full_assessment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validated user-entered vitals/symptoms (optionally blended with a synthetic
    baseline) + knowledge retrieval + educational care + first-aid guidance,
    all in one call. This is what both UIs call on "Analyze"."""
    vitals_result = analyze_user_vitals(payload)
    if vitals_result["status"] != "ok":
        return {"vitals": vitals_result, "knowledge": None, "care": None}

    condition = vitals_result["data"].get("condition") or "general wellness"
    alerts = vitals_result["alerts"]

    knowledge = search_knowledge({"query": f"Educational guidance for {condition} and abnormal vital signs"})
    care = recommend_care({"condition": condition, "alerts": alerts, "patient_id": vitals_result["data"].get("patient_id")})

    return {"vitals": vitals_result, "knowledge": knowledge, "care": care}



# ===========================================================================
# 11. Local demo measurement history (SQLite)
# ===========================================================================
import sqlite3
from pathlib import Path

HISTORY_DB_PATH = Path(os.environ.get("HEALTHCARE_HISTORY_DB", "healthcare_history.sqlite3"))

def _connect_history():
    conn = sqlite3.connect(str(HISTORY_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurement_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at_utc TEXT NOT NULL,
            patient_id INTEGER,
            age INTEGER,
            heart_rate REAL,
            glucose_level REAL,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            spo2 REAL,
            temperature_c REAL,
            symptoms TEXT,
            status_label TEXT,
            alerts_json TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def save_measurement(payload: Dict[str, Any], assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist one user-entered demo measurement; do not pass synthetic baseline-only cases."""
    from json import dumps
    assessment = assessment or {}
    vital_result = assessment.get("vitals", {}) if assessment else {}
    data = vital_result.get("data", {}) if vital_result.get("status") == "ok" else {}
    alerts = vital_result.get("alerts", []) if vital_result else []
    classification = data.get("classification", {})
    recorded_at = datetime.now(timezone.utc).isoformat()
    def chosen(key):
        value = data.get(key)
        return value if value is not None else payload.get(key)

    fields = (
        recorded_at, chosen("patient_id"), chosen("age"),
        chosen("heart_rate"), chosen("glucose_level"),
        chosen("systolic_bp"), chosen("diastolic_bp"),
        chosen("spo2"), chosen("temperature_c"),
        chosen("symptoms"), classification.get("overall_status", "unclassified"),
        dumps(alerts, ensure_ascii=False, default=str),
    )
    try:
        with _connect_history() as conn:
            cur = conn.execute("""
                INSERT INTO measurement_history
                (recorded_at_utc, patient_id, age, heart_rate, glucose_level,
                 systolic_bp, diastolic_bp, spo2, temperature_c, symptoms,
                 status_label, alerts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fields)
            measurement_id = cur.lastrowid
        audit("save_measurement", payload.get("patient_id"), "ok")
        return result("save_measurement", "ok", {"measurement_id": measurement_id, "recorded_at_utc": recorded_at})
    except Exception as exc:
        audit("save_measurement", payload.get("patient_id"), "error")
        return result("save_measurement", "error", {"message": f"Could not save measurement: {exc}"})

def get_measurement_history(limit: int = 50, patient_id: Optional[int] = None) -> Dict[str, Any]:
    """Return recent local demo measurements; newest first."""
    try:
        limit = max(1, min(int(limit), 500))
        with _connect_history() as conn:
            if patient_id is None:
                rows = conn.execute(
                    "SELECT * FROM measurement_history ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM measurement_history WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
                    (int(patient_id), limit)
                ).fetchall()
        items = [dict(row) for row in rows]
        audit("get_measurement_history", patient_id, "ok")
        return result("get_measurement_history", "ok", {"items": items, "count": len(items)})
    except Exception as exc:
        audit("get_measurement_history", patient_id, "error")
        return result("get_measurement_history", "error", {"message": str(exc), "items": []})


# ===========================================================================
# 12. Automated tests
# ===========================================================================
def run_tests() -> str:
    assert get_patient_record({"patient_id": 201})["status"] == "ok"
    assert get_patient_record({"patient_id": 999})["status"] == "not_found"
    assert get_patient_record({"patient_id": -1})["status"] == "error"

    alerts_206 = monitor_vitals({"patient_id": 206})["alerts"]
    assert any(a["type"] == "heart_rate" for a in alerts_206)
    alerts_202 = monitor_vitals({"patient_id": 202})["alerts"]
    assert any(a["type"] == "blood_pressure" for a in alerts_202)
    alerts_212 = monitor_vitals({"patient_id": 212})["alerts"]  # sepsis-like case
    assert any(a["type"] == "temperature" for a in alerts_212)
    assert any(a["type"] == "spo2" for a in alerts_212)

    assert search_knowledge({"query": "high blood pressure"})["status"] == "ok"
    assert search_knowledge({"query": ""})["status"] == "error"

    # User-entered vitals: happy path, validation error, and blended-with-baseline path
    manual = analyze_user_vitals({"heart_rate": 150, "glucose_level": 90, "systolic_bp": 120, "diastolic_bp": 80})
    assert manual["status"] == "ok"
    assert any(a["type"] == "heart_rate" for a in manual["alerts"])

    bad_input = analyze_user_vitals({"heart_rate": 999})
    assert bad_input["status"] == "error"

    empty_input = analyze_user_vitals({})
    assert empty_input["status"] == "error"

    partial_bp = analyze_user_vitals({"systolic_bp": 120})
    assert partial_bp["status"] == "error"
    reversed_bp = analyze_user_vitals({"systolic_bp": 70, "diastolic_bp": 90})
    assert reversed_bp["status"] == "error"

    blended = analyze_user_vitals({"patient_id": 202, "heart_rate": 105})
    assert blended["status"] == "ok"
    assert blended["data"]["used_baseline_patient"] is True
    assert blended["data"]["glucose_level"] == 105  # comes from the synthetic baseline (patient 202)

    emergency = analyze_user_vitals({"heart_rate": 80, "symptoms": "severe chest pain"})
    assert any(a["type"] == "symptom_flag" for a in emergency["alerts"])
    assert emergency["data"]["classification"]["overall_status"] == "emergency"

    # Low readings: low glucose and low blood pressure should both be flagged and labeled
    low_case = analyze_user_vitals({"heart_rate": 55, "glucose_level": 60, "systolic_bp": 85, "diastolic_bp": 55})
    assert low_case["status"] == "ok"
    assert any(a["type"] == "glucose" for a in low_case["alerts"])
    assert any(a["type"] == "blood_pressure" for a in low_case["alerts"])
    assert low_case["data"]["classification"]["glucose_label"] == "Low"
    assert low_case["data"]["classification"]["blood_pressure_label"] == "Low"

    normal_case = analyze_user_vitals({"heart_rate": 72, "glucose_level": 95, "systolic_bp": 118, "diastolic_bp": 76, "spo2": 98, "temperature_c": 36.8})
    assert normal_case["data"]["classification"]["overall_label"].startswith("No configured threshold alerts")
    assert normal_case["data"]["classification"]["overall_status"] == "healthy"

    # Emergency vitals (no symptoms text) should also trigger the emergency banner + first aid
    crit_vitals = analyze_user_vitals({"heart_rate": 140, "spo2": 85, "temperature_c": 40.5})
    assert crit_vitals["data"]["classification"]["overall_status"] == "emergency"
    fa = get_emergency_first_aid(crit_vitals["alerts"])
    assert fa["has_emergency"] is True
    assert len(fa["guidance"]) > 0

    # Care recommendations
    care = recommend_care({"condition": "Type 2 Diabetes", "alerts": []})
    assert care["status"] == "ok"
    assert len(care["data"]["suggestions"]) > 0
    assert "disclaimer" in care["data"]
    assert care["data"]["first_aid"]["has_emergency"] is False

    urgent_care = recommend_care({"condition": "Hypertension", "alerts": [{"type": "blood_pressure", "severity": "critical", "message": "Very high blood-pressure observation"}]})
    assert urgent_care["data"]["escalation_recommended"] is True
    assert urgent_care["data"]["first_aid"]["has_emergency"] is True

    full = run_full_assessment({"patient_id": 201})
    assert full["vitals"]["status"] == "ok"
    assert full["care"]["status"] == "ok"

    full_healthy = run_full_assessment({"patient_id": 203})
    assert full_healthy["vitals"]["data"]["classification"]["overall_status"] == "healthy"

    return "All automated tests passed."


if __name__ == "__main__":
    print(run_tests())
