"""
Unit tests for Personalization, Consent Enforcement, Profile Validation, and RAG.
Tests:
- Consent rejection
- Valid user profile parsing & BMI computation
- Different users produce distinct personalized recommendation inputs/outputs
- RAG retriever guideline groundings
"""

import pytest
from models.schemas import UserProfile
from workflow.graph import run_pipeline
from rag.retriever import retrieve_guidelines


def test_consent_rejection():
    """Verify that workflow refuses to process without explicit user consent."""
    profile_without_consent = {
        "age": 21,
        "height_cm": 170.0,
        "weight_kg": 65.0,
        "fitness_goal": "Improve fitness",
        "activity_level": "Low",
        "diet_preference": "Vegetarian",
        "exercise_preference": "Walking",
        "available_time_mins": 30,
        "allergies": [],
        "optional_health_info": "",
        "consent_given": False  # Consent NOT granted
    }
    
    res = run_pipeline(profile_without_consent)
    final = res.get("final_response", {})
    assert res.get("consent_valid") is False
    assert final.get("status") == "blocked"
    assert "consent" in final.get("message", "").lower()


def test_valid_user_profile():
    """Verify that Pydantic UserProfile model validates inputs and calculates BMI accurately."""
    profile_data = {
        "age": 28,
        "height_cm": 180.0,
        "weight_kg": 75.0,
        "fitness_goal": "Muscle tone",
        "activity_level": "Moderate",
        "diet_preference": "Omnivore",
        "exercise_preference": "Strength Training",
        "available_time_mins": 45,
        "allergies": ["dairy"],
        "optional_health_info": "Desk worker",
        "consent_given": True
    }
    user = UserProfile(**profile_data)
    assert user.age == 28
    # BMI = 75 / (1.8^2) = 23.1
    assert user.bmi == 23.1
    assert user.bmi_category == "Normal weight"


def test_different_users_produce_different_recommendations():
    """Verify that User A and User B receive distinctly tailored recommendation plans."""
    user_a_profile = {
        "age": 21,
        "height_cm": 172.0,
        "weight_kg": 65.0,
        "fitness_goal": "Improve fitness",
        "activity_level": "Low",
        "diet_preference": "Vegetarian",
        "exercise_preference": "Walking",
        "available_time_mins": 30,
        "allergies": [],
        "optional_health_info": "",
        "consent_given": True
    }
    user_a_activity = {
        "user_id": "U001",
        "age": 21,
        "steps_avg": 4200,
        "workout_minutes_avg": 15,
        "sleep_hours_avg": 6.2,
        "resting_heart_rate": 78,
        "activity_level": "Low",
        "workouts_per_week": 1
    }

    user_b_profile = {
        "age": 45,
        "height_cm": 178.0,
        "weight_kg": 80.0,
        "fitness_goal": "General fitness",
        "activity_level": "Moderate",
        "diet_preference": "Vegetarian",
        "exercise_preference": "Cycling",
        "available_time_mins": 60,
        "allergies": [],
        "optional_health_info": "",
        "consent_given": True
    }
    user_b_activity = {
        "user_id": "U002",
        "age": 45,
        "steps_avg": 8500,
        "workout_minutes_avg": 40,
        "sleep_hours_avg": 7.4,
        "resting_heart_rate": 71,
        "activity_level": "Moderate",
        "workouts_per_week": 4
    }

    res_a = run_pipeline(user_a_profile, user_a_activity)["final_response"]
    res_b = run_pipeline(user_b_profile, user_b_activity)["final_response"]

    # User A exercise should focus on walking
    assert "walking" in res_a["exercise"]["title"].lower() or "walking" in res_a["exercise"]["content"].lower()
    
    # User B exercise should focus on cycling
    assert "cycling" in res_b["exercise"]["title"].lower() or "cycling" in res_b["exercise"]["content"].lower()
    
    # Recommendations should not be identical
    assert res_a["exercise"]["content"] != res_b["exercise"]["content"]


def test_rag_retriever_grounding():
    """Verify that ChromaDB retriever returns authoritative guidelines with required metadata."""
    passages = retrieve_guidelines("aerobic physical activity guidelines", top_k=2)
    assert len(passages) >= 1
    first = passages[0]
    assert "source" in first
    assert "topic" in first
    assert "title" in first
    assert "content" in first
    assert len(first["content"]) > 30
