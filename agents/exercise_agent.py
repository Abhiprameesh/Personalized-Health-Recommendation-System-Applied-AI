"""
Exercise Agent: Generates gradual, realistic, personalized exercise recommendations
grounded in authoritative public health guidelines and conditioned on activity metrics.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


from agents.llm_client import invoke_gemini_safely


def generate_exercise_recommendations(
    profile: Dict[str, Any],
    activity: Dict[str, Any],
    guidelines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate tailored exercise plan based on profile, activity history, and RAG guidelines.
    """
    age = profile.get("age", 25)
    goal = profile.get("fitness_goal", "General fitness")
    act_level = profile.get("activity_level", "Moderate")
    exercise_pref = profile.get("exercise_preference", "Walking")
    avail_time = profile.get("available_time_mins", 30)
    
    steps_avg = activity.get("steps_avg", 5000)
    workout_mins = activity.get("workout_minutes_avg", 20)
    workouts_per_week = activity.get("workouts_per_week", 2)
    
    # Grounding context from RAG
    guideline_excerpts = [g.get("content", "") for g in guidelines if "activity" in g.get("topic", "").lower()]
    rag_context = "\n".join(guideline_excerpts[:2])

    prompt = f"""You are a preventive wellness exercise specialist.
User Profile:
- Age: {age}, Goal: {goal}, Activity Level: {act_level}
- Exercise Preference: {exercise_pref}, Available Time: {avail_time} min/day
- Current Activity History: {steps_avg} steps/day, {workout_mins} mins/session, {workouts_per_week} workouts/week.

Authoritative Guidelines (RAG context):
{rag_context}

Instructions:
1. Provide a gradual, realistic physical activity plan tailored to the user's preference ({exercise_pref}) and available time ({avail_time} min).
2. Incorporate progressive overload over 2 to 4 weeks.
3. DO NOT diagnose medical conditions or prescribe treatment.
4. DO NOT recommend medications.
5. Provide a 4-line summary including weekly frequency, session duration, and progression.
"""

    llm_text, model_used = invoke_gemini_safely(prompt)
    if llm_text:
        return {
            "summary": f"Personalized {exercise_pref}-Centered Progression Plan ({goal})",
            "routine": llm_text,
            "target_minutes_per_session": min(avail_time, 45 if act_level == "Low" else 60),
            "frequency_per_week": 4 if act_level in ["Low", "Moderate"] else 5,
            "focus_area": exercise_pref,
            "mode": f"Gemini ({model_used})"
        }

    # High-quality deterministic generator grounded in WHO/CDC guidelines & user parameters
    if exercise_pref.lower() == "walking" or act_level.lower() == "low":
        target_mins = min(avail_time, 30)
        routine = (
            f"• **Baseline Phase (Weeks 1-2):** 5 days/week of {target_mins}-minute brisk walking at an aerobic conversational pace.\n"
            f"• **Target:** Gradually elevate daily steps from baseline ({steps_avg:,}) toward 7,000-8,000 steps.\n"
            f"• **Progression Phase (Weeks 3-4):** Introduce 10-minute interval walks (3 minutes brisk, 1 minute recovery pace) twice weekly.\n"
            f"• **Mobility & Strength Add-on:** 2 weekly 10-minute bodyweight mobility sessions (chair squats, calf raises, wall push-ups)."
        )
    elif exercise_pref.lower() == "cycling":
        target_mins = min(avail_time, 50 if act_level == "Moderate" else 35)
        freq = 4 if act_level == "Moderate" else 3
        routine = (
            f"• **Endurance Building:** {freq} days/week of moderate-intensity cycling sessions ({target_mins} minutes per ride).\n"
            f"• **Cadence & Pacing:** Maintain steady cadence (70-85 RPM) focusing on Zone 2 aerobic endurance.\n"
            f"• **Progression & Recovery:** Alternate between steady flat rides and moderate incline resistance, reserving 2 dedicated rest/active-recovery days per week.\n"
            f"• **Supplementary Routine:** Integrate 15-minute core and lower-body stability routines twice weekly."
        )
    else:
        target_mins = min(avail_time, 40)
        routine = (
            f"• **Structured Routine:** 3 to 4 days/week of {exercise_pref} sessions lasting {target_mins} minutes.\n"
            f"• **Warm-up & Cool-down:** 5 minutes dynamic warm-up and 5 minutes post-session flexibility.\n"
            f"• **Progressive Adaptation:** Gradual 10% increase in weekly volume every fortnight, adhering to WHO 150-300 min aerobic recommendations.\n"
            f"• **Rest Cycle:** Scheduled recovery days to prevent fatigue and support muscular adaptation."
        )

    return {
        "summary": f"Gradual {exercise_pref} Plan for {goal} (Level: {act_level})",
        "routine": routine,
        "target_minutes_per_session": min(avail_time, 45),
        "frequency_per_week": 4 if act_level == "Low" else 4,
        "focus_area": exercise_pref,
        "mode": "Authoritative Guidelines Grounded"
    }
