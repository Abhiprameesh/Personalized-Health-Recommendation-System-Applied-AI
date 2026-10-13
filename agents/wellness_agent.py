"""
Preventive Wellness Agent: Generates restorative sleep, hydration, and daily lifestyle routine
recommendations conditioned on activity metrics (sleep hours, resting heart rate).
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from safety.validator import check_concerning_symptoms, ESCALATION_NOTICE

load_dotenv()


from agents.llm_client import invoke_gemini_safely


def generate_wellness_recommendations(
    profile: Dict[str, Any],
    activity: Dict[str, Any],
    guidelines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate lifestyle, sleep hygiene, and recovery routines.
    Immediately flags any self-reported acute symptoms.
    """
    health_info = profile.get("optional_health_info", "")
    
    # Check if user reported any concerning symptom
    symptom_issues = check_concerning_symptoms(health_info)
    if symptom_issues:
        return {
            "summary": "Medical Escalation Required",
            "routine": ESCALATION_NOTICE,
            "escalation": True,
            "issues": symptom_issues,
            "mode": "Safety-Escalation"
        }

    sleep_hours = activity.get("sleep_hours_avg", 7.0)
    resting_hr = activity.get("resting_heart_rate", 72)
    age = profile.get("age", 25)

    # Grounding context from RAG
    guideline_excerpts = [
        g.get("content", "") for g in guidelines 
        if "sleep" in g.get("topic", "").lower() or "preventive" in g.get("topic", "").lower()
    ]
    rag_context = "\n".join(guideline_excerpts[:2])

    prompt = f"""You are a preventive lifestyle wellness specialist.
User Profile:
- Age: {age}
- Activity Metrics: Average Sleep: {sleep_hours} hrs/night, Resting Heart Rate: {resting_hr} bpm.

Authoritative Sleep and Preventive Guidelines (RAG context):
{rag_context}

Instructions:
1. Provide actionable sleep hygiene and daily preventive routine suggestions.
2. If sleep is below 7 hours, emphasize circadian regularity and screen wind-down rituals.
3. If resting heart rate is elevated (>75 bpm), emphasize parasympathetic breathing and hydration.
4. DO NOT diagnose any medical condition or prescribe medication.
5. Provide a concise 4-bullet routine for daily habits and recovery.
"""

    llm_text, model_used = invoke_gemini_safely(prompt)
    if llm_text:
        return {
            "summary": "Restorative Sleep & Daily Preventive Routine",
            "routine": llm_text,
            "sleep_hours_avg": sleep_hours,
            "resting_heart_rate": resting_hr,
            "escalation": False,
            "mode": f"Gemini ({model_used})"
        }

    # High-quality deterministic generator grounded in Sleep Foundation & AHA guidelines
    sleep_nudge = ""
    if sleep_hours < 7.0:
        sleep_nudge = (
            f"• **Sleep Duration Extension:** Your current baseline is {sleep_hours} hours. Aim to incrementally add 20 to 30 minutes "
            "by setting a fixed bedtime 30 minutes earlier, moving toward the recommended 7 to 9-hour restorative window."
        )
    else:
        sleep_nudge = (
            f"• **Sleep Regularity:** Your average sleep ({sleep_hours} hrs) is within the healthy adult range. "
            "Anchor your circadian rhythm by waking and sleeping at consistent times (±30 mins), even on weekends."
        )

    rhr_nudge = ""
    if resting_hr > 75:
        rhr_nudge = (
            f"• **Recovery & Stress Regulation:** Resting heart rate is currently {resting_hr} bpm. Incorporate 5 to 10 minutes of daily "
            "diaphragmatic box breathing (4s inhale, 4s hold, 4s exhale, 4s hold) to activate parasympathetic recovery."
        )
    else:
        rhr_nudge = (
            f"• **Cardiovascular Baseline:** Resting heart rate ({resting_hr} bpm) indicates good baseline cardiovascular recovery. "
            "Maintain baseline hydration (2 to 2.5 L water) and include light mobility breaks every hour of seated work."
        )

    routine = (
        f"{sleep_nudge}\n"
        f"{rhr_nudge}\n"
        f"• **Evening Digital Sunset:** Dim room lights and power down bright screens 45 to 60 minutes before bed to support melatonin onset.\n"
        f"• **Morning Light Exposure:** Step outside for 10 to 15 minutes of natural sunlight within one hour of waking to calibrate alertness."
    )

    return {
        "summary": "Daily Preventive Wellness & Circadian Health",
        "routine": routine,
        "sleep_hours_avg": sleep_hours,
        "resting_heart_rate": resting_hr,
        "escalation": False,
        "mode": "Authoritative Guidelines Grounded"
    }
