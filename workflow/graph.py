import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from models.schemas import UserProfile, ActivityData, WorkflowState
from rag.retriever import retrieve_guidelines
from agents.exercise_agent import generate_exercise_recommendations
from agents.nutrition_agent import generate_nutrition_recommendations
from agents.wellness_agent import generate_wellness_recommendations
from safety.validator import validate_recommendations, ESCALATION_NOTICE

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "activity_data.csv"

# Disclaimer required by technical and medical guidelines
DISCLAIMER_TEXT = (
    "This prototype provides general wellness information and is not a medical diagnostic or treatment system. "
    "RAG helps reduce unsupported claims by grounding recommendations in retrieved knowledge, but it does not guarantee factual correctness. "
    "Always consult a qualified healthcare professional before beginning any new exercise or dietary regimen."
)


def node_build_profile(state: WorkflowState) -> Dict[str, Any]:
    """Validate user profile and verify consent."""
    profile_data = state.get("user_profile", {})
    consent = profile_data.get("consent_given", False)
    
    if not consent:
        return {
            "consent_valid": False,
            "error": "Consent not provided. The application must refuse to process personalized data without explicit consent.",
            "final_response": {
                "status": "blocked",
                "message": "Processing halted: User consent is mandatory before personalized wellness recommendations can be generated."
            }
        }
        
    try:
        # Validate through Pydantic
        validated_profile = UserProfile(**profile_data)
        profile_dict = validated_profile.model_dump()
        profile_dict["bmi"] = validated_profile.bmi
        profile_dict["bmi_category"] = validated_profile.bmi_category
        return {"user_profile": profile_dict, "consent_valid": True, "error": None}
    except Exception as e:
        return {
            "consent_valid": True,
            "error": f"Invalid profile data: {str(e)}",
            "final_response": {"status": "error", "message": f"Profile validation error: {str(e)}"}
        }


def node_load_activity_data(state: WorkflowState) -> Dict[str, Any]:
    """Associate synthetic user activity profile or validate manually provided activity data."""
    if state.get("error") or not state.get("consent_valid"):
        return {}

    raw_act = state.get("activity_data", {})
    user_id = raw_act.get("user_id", "")
    
    # If user selected a synthetic user ID and file exists, load from CSV
    if user_id and DATA_CSV.exists():
        try:
            df = pd.read_csv(DATA_CSV)
            match = df[df["user_id"] == user_id]
            if not match.empty:
                row = match.iloc[0].to_dict()
                return {"activity_data": row}
        except Exception as e:
            print(f"Warning: could not lookup synthetic user {user_id}: {e}")
            
    # Default/fallback activity profile
    act_dict = {
        "user_id": raw_act.get("user_id", "CustomUser"),
        "age": raw_act.get("age", state.get("user_profile", {}).get("age", 25)),
        "steps_avg": raw_act.get("steps_avg", 5000),
        "workout_minutes_avg": raw_act.get("workout_minutes_avg", 25),
        "sleep_hours_avg": raw_act.get("sleep_hours_avg", 7.0),
        "resting_heart_rate": raw_act.get("resting_heart_rate", 72),
        "activity_level": raw_act.get("activity_level", state.get("user_profile", {}).get("activity_level", "Moderate")),
        "workouts_per_week": raw_act.get("workouts_per_week", 3)
    }
    return {"activity_data": act_dict}


def node_retrieve_guidelines(state: WorkflowState) -> Dict[str, Any]:
    """Retrieve authoritative guidelines from ChromaDB RAG vector store."""
    if state.get("error") or not state.get("consent_valid"):
        return {}

    profile = state.get("user_profile", {})
    ex_pref = profile.get("exercise_preference", "walking")
    goal = profile.get("fitness_goal", "general fitness")
    diet = profile.get("diet_preference", "vegetarian")
    
    # Retrieve across relevant categories
    act_query = f"{ex_pref} aerobic physical activity {goal} duration frequency progression"
    nut_query = f"{diet} balanced nutrition protein intake hydration healthy eating"
    well_query = "sleep hygiene circadian rhythm resting heart rate recovery stress prevention"
    
    docs_act = retrieve_guidelines(act_query, top_k=2, topic="physical activity")
    docs_nut = retrieve_guidelines(nut_query, top_k=2, topic="nutrition")
    docs_well = retrieve_guidelines(well_query, top_k=2, topic="sleep wellness")
    
    combined_docs = docs_act + docs_nut + docs_well
    return {"retrieved_context": combined_docs}


def node_exercise_agent(state: WorkflowState) -> Dict[str, Any]:
    """Execute specialized Exercise Agent."""
    if state.get("error") or not state.get("consent_valid"):
        return {}
    profile = state.get("user_profile", {})
    activity = state.get("activity_data", {})
    context = state.get("retrieved_context", [])
    rec = generate_exercise_recommendations(profile, activity, context)
    return {"exercise_recommendation": rec}


def node_nutrition_agent(state: WorkflowState) -> Dict[str, Any]:
    """Execute specialized Nutrition Agent."""
    if state.get("error") or not state.get("consent_valid"):
        return {}
    profile = state.get("user_profile", {})
    activity = state.get("activity_data", {})
    context = state.get("retrieved_context", [])
    rec = generate_nutrition_recommendations(profile, activity, context)
    return {"nutrition_recommendation": rec}


def node_wellness_agent(state: WorkflowState) -> Dict[str, Any]:
    """Execute specialized Preventive Wellness Agent."""
    if state.get("error") or not state.get("consent_valid"):
        return {}
    profile = state.get("user_profile", {})
    activity = state.get("activity_data", {})
    context = state.get("retrieved_context", [])
    rec = generate_wellness_recommendations(profile, activity, context)
    return {"wellness_recommendation": rec}


def node_safety_validator(state: WorkflowState) -> Dict[str, Any]:
    """Execute Safety Validation Layer across inputs and generated outputs."""
    if state.get("error") or not state.get("consent_valid"):
        return {}

    profile = state.get("user_profile", {})
    health_info = profile.get("optional_health_info", "")
    allergies = profile.get("allergies", [])
    
    ex_rec = state.get("exercise_recommendation", {}).get("routine", "")
    nut_rec = state.get("nutrition_recommendation", {}).get("meal_plan", "")
    well_rec = state.get("wellness_recommendation", {}).get("routine", "")
    
    safety_check = validate_recommendations(
        user_input_text=health_info,
        exercise_rec=ex_rec,
        nutrition_rec=nut_rec,
        wellness_rec=well_rec,
        user_allergies=allergies
    )
    return {"safety_result": safety_check}


def node_final_formatter(state: WorkflowState) -> Dict[str, Any]:
    """Assemble final personalized recommendations with citations, confidence, and disclaimers."""
    if state.get("error") or not state.get("consent_valid"):
        return {"final_response": state.get("final_response", {})}

    safety = state.get("safety_result", {})
    sanitized = safety.get("sanitized_output", {})
    context = state.get("retrieved_context", [])
    profile = state.get("user_profile", {})
    activity = state.get("activity_data", {})

    # Extract unique knowledge sources cited
    sources_used = []
    seen = set()
    for doc in context:
        key = (doc.get("source"), doc.get("topic"), doc.get("title"))
        if key not in seen:
            seen.add(key)
            sources_used.append({
                "source": doc.get("source"),
                "topic": doc.get("topic"),
                "title": doc.get("title"),
                "sample_passage": doc.get("content", "")[:140] + "..."
            })

    # Prepare formatted payload
    final_output = {
        "status": "success",
        "user_summary": {
            "age": profile.get("age"),
            "bmi": profile.get("bmi"),
            "bmi_category": profile.get("bmi_category"),
            "goal": profile.get("fitness_goal"),
            "activity_baseline": activity.get("activity_level"),
            "daily_steps": activity.get("steps_avg"),
            "resting_hr": activity.get("resting_heart_rate"),
            "diet": profile.get("diet_preference"),
            "allergies": profile.get("allergies", [])
        },
        "exercise": {
            "title": state.get("exercise_recommendation", {}).get("summary", "Physical Activity Routine"),
            "content": sanitized.get("exercise", state.get("exercise_recommendation", {}).get("routine", ""))
        },
        "nutrition": {
            "title": state.get("nutrition_recommendation", {}).get("summary", "Nutritional Guidance"),
            "content": sanitized.get("nutrition", state.get("nutrition_recommendation", {}).get("meal_plan", ""))
        },
        "preventive_wellness": {
            "title": state.get("wellness_recommendation", {}).get("summary", "Preventive Wellness Habits"),
            "content": sanitized.get("wellness", state.get("wellness_recommendation", {}).get("routine", ""))
        },
        "safety_audit": {
            "status": safety.get("status", "safe"),
            "action": safety.get("action", "show"),
            "issues_detected": safety.get("issues", []),
            "escalation_message": safety.get("escalation_message")
        },
        "knowledge_sources": sources_used,
        "disclaimer": DISCLAIMER_TEXT
    }

    return {"final_response": final_output}


def router_consent_check(state: WorkflowState) -> str:
    """Conditional edge: check if consent was granted and profile is valid."""
    if not state.get("consent_valid", False) or state.get("error"):
        return END
    return "load_activity_data"


def build_recommendation_graph():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("build_profile", node_build_profile)
    workflow.add_node("load_activity_data", node_load_activity_data)
    workflow.add_node("retrieve_guidelines", node_retrieve_guidelines)
    workflow.add_node("exercise_agent", node_exercise_agent)
    workflow.add_node("nutrition_agent", node_nutrition_agent)
    workflow.add_node("wellness_agent", node_wellness_agent)
    workflow.add_node("safety_validator", node_safety_validator)
    workflow.add_node("final_formatter", node_final_formatter)
    
    # Define linear & conditional edges
    workflow.add_edge(START, "build_profile")
    workflow.add_conditional_edges(
        "build_profile",
        router_consent_check,
        {
            "load_activity_data": "load_activity_data",
            END: END
        }
    )
    workflow.add_edge("load_activity_data", "retrieve_guidelines")
    workflow.add_edge("retrieve_guidelines", "exercise_agent")
    workflow.add_edge("exercise_agent", "nutrition_agent")
    workflow.add_edge("nutrition_agent", "wellness_agent")
    workflow.add_edge("wellness_agent", "safety_validator")
    workflow.add_edge("safety_validator", "final_formatter")
    workflow.add_edge("final_formatter", END)
    
    return workflow.compile()


# Global compiled app
app_workflow = build_recommendation_graph()


def run_pipeline(profile_dict: Dict[str, Any], activity_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Entry point function to execute the LangGraph pipeline."""
    initial_state: WorkflowState = {
        "user_profile": profile_dict,
        "activity_data": activity_dict or {},
        "retrieved_context": [],
        "exercise_recommendation": {},
        "nutrition_recommendation": {},
        "wellness_recommendation": {},
        "safety_result": {},
        "final_response": {},
        "consent_valid": False,
        "error": None
    }
    result = app_workflow.invoke(initial_state)
    return result
