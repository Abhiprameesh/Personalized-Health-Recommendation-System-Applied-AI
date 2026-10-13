"""
Personalized Health Recommendation System: AI-Based Design for Preventive Wellness
Applied AI Case Study 19 (22AI741)
Streamlit Dashboard Prototype
"""

import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup path for internal imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from workflow.graph import run_pipeline, DISCLAIMER_TEXT
from safety.validator import ESCALATION_NOTICE

# Page Configuration
st.set_page_config(
    page_title="Personalized Health AI - Preventive Wellness",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich, polished visual aesthetics
st.markdown("""
<style>
    /* Metric and Card Styling */
    .health-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background-color: #e0f2fe; color: #0369a1; }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-amber { background-color: #fef3c7; color: #b45309; }
    .badge-red { background-color: #fee2e2; color: #b91c1c; }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 10px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load synthetic activity data for quick lookup
DATA_PATH = ROOT_DIR / "data" / "activity_data.csv"
activity_df = pd.DataFrame()
if DATA_PATH.exists():
    try:
        activity_df = pd.read_csv(DATA_PATH)
    except Exception:
        pass


def reset_session():
    """Clear session state values."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def load_user_a():
    """Pre-load Demo Scenario 1: User A."""
    st.session_state["age"] = 21
    st.session_state["height"] = 172.0
    st.session_state["weight"] = 65.0
    st.session_state["goal"] = "Improve fitness"
    st.session_state["act_level"] = "Low"
    st.session_state["diet"] = "Vegetarian"
    st.session_state["exercise"] = "Walking"
    st.session_state["time"] = 30
    st.session_state["allergies"] = "peanuts"
    st.session_state["health_info"] = "No known medical issues; looking to build routine."
    st.session_state["selected_user_id"] = "U001"
    st.session_state["consent"] = True
    st.session_state["auto_trigger"] = False


def load_user_b():
    """Pre-load Demo Scenario 2: User B."""
    st.session_state["age"] = 45
    st.session_state["height"] = 178.0
    st.session_state["weight"] = 80.0
    st.session_state["goal"] = "General fitness"
    st.session_state["act_level"] = "Moderate"
    st.session_state["diet"] = "Vegetarian"
    st.session_state["exercise"] = "Cycling"
    st.session_state["time"] = 45
    st.session_state["allergies"] = ""
    st.session_state["health_info"] = "Enjoys outdoor cycling on weekends; wants endurance progression."
    st.session_state["selected_user_id"] = "U002"
    st.session_state["consent"] = True
    st.session_state["auto_trigger"] = False


# Initialize default session state if empty
if "age" not in st.session_state:
    st.session_state["age"] = 25
if "height" not in st.session_state:
    st.session_state["height"] = 170.0
if "weight" not in st.session_state:
    st.session_state["weight"] = 68.0
if "goal" not in st.session_state:
    st.session_state["goal"] = "Improve fitness"
if "act_level" not in st.session_state:
    st.session_state["act_level"] = "Low"
if "diet" not in st.session_state:
    st.session_state["diet"] = "Vegetarian"
if "exercise" not in st.session_state:
    st.session_state["exercise"] = "Walking"
if "time" not in st.session_state:
    st.session_state["time"] = 30
if "allergies" not in st.session_state:
    st.session_state["allergies"] = ""
if "health_info" not in st.session_state:
    st.session_state["health_info"] = ""
if "selected_user_id" not in st.session_state:
    st.session_state["selected_user_id"] = "U001"
if "consent" not in st.session_state:
    st.session_state["consent"] = True

# Header
st.title("🌿 Personalized Health AI")
st.markdown("**AI-Based Design for Preventive Wellness** | *Case Study 19 — Applied AI (22AI741)*")
st.caption("Educational proof-of-concept demonstrating LangGraph multi-agent orchestration, RAG grounding, safety boundaries, and consent controls.")

# Top Action Toolbar for Viva / Demos
with st.container():
    col_btn1, col_btn2, col_btn3, col_info = st.columns([1.5, 1.5, 1.2, 4])
    with col_btn1:
        if st.button("👤 Load Example User A", use_container_width=True, help="Age 21, Low Activity, Walking, Vegetarian"):
            load_user_a()
            st.rerun()
    with col_btn2:
        if st.button("🚴 Load Example User B", use_container_width=True, help="Age 45, Moderate Activity, Cycling, Vegetarian"):
            load_user_b()
            st.rerun()
    with col_btn3:
        if st.button("🔄 Clear Session", use_container_width=True):
            reset_session()
    with col_info:
        st.info("💡 **Demo Quick-Load:** Click 'User A' or 'User B' to populate case study inputs, then click **Generate Recommendations**.")

st.markdown("---")

# Main 2-Column Layout: Inputs (Left) and Output Dashboards (Right)
col_input, col_output = st.columns([1.1, 1.9], gap="large")

# ==============================================================================
# SECTION 1 & 2: USER PROFILE & ACTIVITY DATA INPUTS
# ==============================================================================
with col_input:
    st.markdown('<div class="section-header">📋 1. User Profile</div>', unsafe_allow_html=True)
    
    col_age, col_h, col_w = st.columns(3)
    with col_age:
        age_val = st.number_input("Age", min_value=15, max_value=95, value=st.session_state["age"], step=1)
    with col_h:
        height_val = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=st.session_state["height"], step=1.0)
    with col_w:
        weight_val = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=st.session_state["weight"], step=0.5)

    # Compute quick BMI badge
    bmi_calc = round(weight_val / ((height_val / 100.0) ** 2), 1)
    st.markdown(f"**Computed BMI:** `{bmi_calc}` ({'Normal weight' if 18.5 <= bmi_calc < 25 else 'Underweight' if bmi_calc < 18.5 else 'Overweight'})")

    col_g, col_act = st.columns(2)
    goal_options = ["Improve fitness", "General fitness", "Muscle tone", "Cardiovascular health", "Healthy weight management"]
    with col_g:
        goal_val = st.selectbox("Fitness Goal", options=goal_options, index=goal_options.index(st.session_state["goal"]) if st.session_state["goal"] in goal_options else 0)
    with col_act:
        act_levels = ["Low", "Moderate", "High"]
        act_val = st.selectbox("Activity Level", options=act_levels, index=act_levels.index(st.session_state["act_level"]) if st.session_state["act_level"] in act_levels else 0)

    col_diet, col_ex = st.columns(2)
    diet_options = ["Vegetarian", "Vegan", "Omnivore", "Pescatarian", "Low-Carb"]
    with col_diet:
        diet_val = st.selectbox("Diet Preference", options=diet_options, index=diet_options.index(st.session_state["diet"]) if st.session_state["diet"] in diet_options else 0)
    with col_ex:
        ex_options = ["Walking", "Cycling", "Running", "Swimming", "Home Workout", "Strength Training"]
        ex_val = st.selectbox("Exercise Preference", options=ex_options, index=ex_options.index(st.session_state["exercise"]) if st.session_state["exercise"] in ex_options else 0)

    time_val = st.slider("Available Time per Day (minutes)", min_value=15, max_value=120, value=st.session_state["time"], step=5)
    
    allergies_val = st.text_input(
        "Food Allergies / Intolerances (comma-separated)",
        value=st.session_state["allergies"],
        placeholder="e.g. peanuts, dairy, gluten, shellfish"
    )

    health_info_val = st.text_area(
        "Optional Self-Reported Health Info / Symptoms",
        value=st.session_state["health_info"],
        placeholder="e.g. Desk worker, sedentary during the day. (Try entering 'chest pain' to test Demo 5 safety escalation)",
        help="Entering concerning clinical symptoms triggers mandatory safety escalation."
    )

    st.markdown('<div class="section-header">📊 2. Activity Data</div>', unsafe_allow_html=True)
    
    data_mode = st.radio("Activity Data Source", ["Synthetic Profile from Dataset (300 Users)", "Manual Input"], horizontal=True)

    loaded_activity = {}
    if data_mode.startswith("Synthetic"):
        user_ids = activity_df["user_id"].tolist() if not activity_df.empty else ["U001", "U002", "U003"]
        default_idx = user_ids.index(st.session_state["selected_user_id"]) if st.session_state["selected_user_id"] in user_ids else 0
        selected_uid = st.selectbox("Select Synthetic User ID", options=user_ids, index=default_idx)
        
        if not activity_df.empty and selected_uid in activity_df["user_id"].values:
            row = activity_df[activity_df["user_id"] == selected_uid].iloc[0]
            loaded_activity = row.to_dict()
            st.caption(f"📁 Loaded Synthetic Record: `{selected_uid}` | Avg Steps: **{loaded_activity['steps_avg']:,}** | Sleep: **{loaded_activity['sleep_hours_avg']}h** | RHR: **{loaded_activity['resting_heart_rate']} bpm**")
    else:
        c1, c2 = st.columns(2)
        with c1:
            m_steps = st.number_input("Average Daily Steps", min_value=1000, max_value=30000, value=5000, step=500)
            m_sleep = st.number_input("Average Sleep (hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
        with c2:
            m_mins = st.number_input("Workout Minutes / Day", min_value=0, max_value=180, value=25, step=5)
            m_rhr = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=140, value=72, step=1)
        
        loaded_activity = {
            "user_id": "ManualUser",
            "age": age_val,
            "steps_avg": m_steps,
            "workout_minutes_avg": m_mins,
            "sleep_hours_avg": m_sleep,
            "resting_heart_rate": m_rhr,
            "activity_level": act_val,
            "workouts_per_week": 3
        }

    st.markdown('<div class="section-header">🔒 3. Consent & Privacy</div>', unsafe_allow_html=True)
    consent_checked = st.checkbox(
        "I consent to using the information I provide to generate personalized wellness recommendations.",
        value=st.session_state["consent"],
        help="Demonstrates responsible AI data governance. The application will refuse processing without explicit opt-in."
    )

    st.markdown('<div class="section-header">⚡ 4. Generate Recommendations</div>', unsafe_allow_html=True)
    generate_clicked = st.button("🚀 Generate Recommendations", type="primary", use_container_width=True)


# ==============================================================================
# PIPELINE EXECUTION & RESULTS DISPLAY
# ==============================================================================
with col_output:
    if generate_clicked:
        # Check consent first (DEMO 6)
        if not consent_checked:
            st.error("⛔ **Consent Rejected (Demo 6):** User consent was not granted. The system strictly refuses to process personalized data without explicit consent.")
            st.warning("Please check the consent box in Section 3 to proceed.")
        else:
            # Parse allergies list
            allergy_list = [a.strip() for a in allergies_val.split(",") if a.strip()]

            profile_payload = {
                "age": age_val,
                "height_cm": height_val,
                "weight_kg": weight_val,
                "fitness_goal": goal_val,
                "activity_level": act_val,
                "diet_preference": diet_val,
                "exercise_preference": ex_val,
                "available_time_mins": time_val,
                "allergies": allergy_list,
                "optional_health_info": health_info_val,
                "consent_given": consent_checked
            }

            with st.spinner("🤖 Orchestrating agents via LangGraph (Exercise, Nutrition, Wellness) & Grounding with ChromaDB RAG..."):
                result = run_pipeline(profile_payload, loaded_activity)

            final_resp = result.get("final_response", {})

            # Store in session state for tabs
            st.session_state["last_result"] = final_resp

    if "last_result" in st.session_state:
        final_resp = st.session_state["last_result"]
        safety_audit = final_resp.get("safety_audit", {})
        is_safe = safety_audit.get("status") == "safe"
        issues = safety_audit.get("issues_detected", [])

        # ==============================================================================
        # SECTION 7: SAFETY CHECK (Displayed at Top for Clinical Vigilance)
        # ==============================================================================
        st.markdown('<div class="section-header">🛡️ 7. Safety & Clinical Boundary Audit</div>', unsafe_allow_html=True)
        
        if is_safe:
            st.success("✅ **Safety Check: PASSED** — All recommendations comply with preventive wellness scope.")
            with st.expander("🔍 View Safety Audit Details", expanded=False):
                st.markdown("""
                - **Diagnostic Language:** `PASSED` (No clinical diagnoses made)
                - **Medication & Prescriptions:** `PASSED` (No pharmaceutical advice)
                - **Allergen Verification:** `PASSED` (Declared food allergens strictly excluded)
                - **Emergency Red-Flag Check:** `PASSED` (No acute symptoms detected)
                """)
        else:
            st.error("⚠️ **Safety Check: ALERT TRIGGERED** — Action: Medical Escalation")
            st.error(f"**Clinical Boundary Notice:** {safety_audit.get('escalation_message', ESCALATION_NOTICE)}")
            if issues:
                with st.expander("⚠️ Triggered Safety Reasons", expanded=True):
                    for iss in issues:
                        st.write(f"- {iss}")

        # Activity Baseline Metrics Row
        act_summary = final_resp.get("user_summary", {})
        if act_summary:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Daily Steps", f"{act_summary.get('daily_steps', 0):,}")
            m2.metric("Resting Heart Rate", f"{act_summary.get('resting_hr', 0)} bpm")
            m3.metric("BMI Category", f"{act_summary.get('bmi_category')}")
            m4.metric("Activity Baseline", f"{act_summary.get('activity_baseline')}")

        # ==============================================================================
        # SECTION 5: RECOMMENDATIONS
        # ==============================================================================
        st.markdown('<div class="section-header">💡 5. Personalized Recommendations</div>', unsafe_allow_html=True)

        tab_ex, tab_nut, tab_well = st.tabs(["🏃 Exercise Agent", "🥗 Nutrition Agent", "🌙 Preventive Wellness Agent"])

        with tab_ex:
            ex_data = final_resp.get("exercise", {})
            st.subheader(ex_data.get("title", "Physical Activity Routine"))
            st.markdown(ex_data.get("content", "No exercise data available."))

        with tab_nut:
            nut_data = final_resp.get("nutrition", {})
            st.subheader(nut_data.get("title", "Nutritional Guidance"))
            st.markdown(nut_data.get("content", "No nutrition data available."))
            if act_summary.get("allergies"):
                st.success(f"🛡️ **Allergen Filter Active:** Excluded ingredients matching: `{', '.join(act_summary.get('allergies'))}`")

        with tab_well:
            well_data = final_resp.get("preventive_wellness", {})
            st.subheader(well_data.get("title", "Preventive Wellness Habits"))
            st.markdown(well_data.get("content", "No wellness data available."))

        # ==============================================================================
        # SECTION 6: KNOWLEDGE SOURCES (RAG)
        # ==============================================================================
        st.markdown('<div class="section-header">📚 6. Knowledge Sources Used (RAG Grounding)</div>', unsafe_allow_html=True)
        sources = final_resp.get("knowledge_sources", [])
        
        st.caption("ℹ️ **RAG Transparency Note:** *RAG helps reduce unsupported claims by grounding recommendations in retrieved knowledge, but it does not guarantee factual correctness.*")
        
        if sources:
            for s in sources:
                with st.expander(f"📖 [{s.get('source')}] {s.get('title')} ({s.get('topic')})"):
                    st.markdown(f"**Source Organization:** `{s.get('source')}`")
                    st.markdown(f"**Topic Category:** `{s.get('topic')}`")
                    st.markdown(f"**Retrieved Evidence Passage:**\n> {s.get('sample_passage')}")
        else:
            st.write("No knowledge passages retrieved.")

        # Medical Disclaimer Banner
        st.markdown("---")
        st.caption(f"⚖️ **Medical Disclaimer:** {final_resp.get('disclaimer', DISCLAIMER_TEXT)}")

    else:
        # Welcome placeholder when no recommendation generated yet
        st.info("👈 **Get Started:** Review the profile inputs on the left, verify consent, and click **Generate Recommendations** to run the LangGraph workflow.")
        
        st.markdown("""
        ### Case Study Architecture Flow:
        1. **User Profile & Activity Data:** Aggregates user inputs with synthetic wearable metrics.
        2. **LangGraph Multi-Agent Orchestration:**
           - `Exercise Agent`: Generates realistic, gradual activity progressions.
           - `Nutrition Agent`: Delivers diet-tailored templates with strict allergen safeguards.
           - `Preventive Wellness Agent`: Focuses on sleep hygiene, circadian alignment, and hydration.
        3. **ChromaDB RAG Grounding:** Retrieves authoritative public health guidance (WHO, CDC, AHA, Sleep Foundation).
        4. **Safety & Medical Boundary Layer:** Enforces non-diagnostic policies and clinical escalation.
        """)
