"""
MCP server for Smart Healthcare Assistant.
Run locally with: python healthcare_mcp_server.py
Connect this stdio server from an MCP-compatible host. Do not expose real patient data
through an untrusted host or public demo.
"""
from typing import Optional
from mcp.server.fastmcp import FastMCP
from healthcare_core import (
    analyze_user_vitals,
    save_measurement,
    get_measurement_history,
    search_knowledge,
)

mcp = FastMCP("Smart Healthcare Assistant")

@mcp.tool()
def analyze_vitals(
    heart_rate: Optional[float] = None,
    systolic_bp: Optional[int] = None,
    diastolic_bp: Optional[int] = None,
    spo2: Optional[float] = None,
    temperature_c: Optional[float] = None,
    glucose_level: Optional[float] = None,
    age: Optional[int] = None,
    symptoms: Optional[str] = None,
    known_condition: Optional[str] = None,
) -> dict:
    """Validate and summarize user-entered vital signs; returns educational flags, not a diagnosis."""
    payload = {
        "heart_rate": heart_rate, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "spo2": spo2, "temperature_c": temperature_c, "glucose_level": glucose_level,
        "age": age, "symptoms": symptoms, "known_condition": known_condition,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return analyze_user_vitals(payload)

@mcp.tool()
def save_vitals(
    heart_rate: Optional[float] = None,
    systolic_bp: Optional[int] = None,
    diastolic_bp: Optional[int] = None,
    spo2: Optional[float] = None,
    temperature_c: Optional[float] = None,
    glucose_level: Optional[float] = None,
    age: Optional[int] = None,
    symptoms: Optional[str] = None,
) -> dict:
    """Validate then save user-entered readings to the local SQLite demo history."""
    payload = {
        "heart_rate": heart_rate, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "spo2": spo2, "temperature_c": temperature_c, "glucose_level": glucose_level,
        "age": age, "symptoms": symptoms,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    checked = analyze_user_vitals(payload)
    if checked.get("status") != "ok":
        return checked
    saved = save_measurement(payload, {"vitals": checked})
    return {"validation": checked, "save": saved}

@mcp.tool()
def get_vitals_history(limit: int = 20) -> dict:
    """Retrieve recent locally stored demo readings, newest first."""
    return get_measurement_history(limit=limit)

@mcp.tool()
def search_health_knowledge(query: str, n_results: int = 3) -> dict:
    """Search the app's small educational knowledge base; not a substitute for clinical advice."""
    return search_knowledge({"query": query, "n_results": n_results})

if __name__ == "__main__":
    mcp.run()
