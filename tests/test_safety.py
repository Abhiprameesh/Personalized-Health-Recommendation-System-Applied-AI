"""
Unit tests for Safety Validation Layer.
Tests:
- Diagnostic language detection
- Medication / prescription language detection
- Allergy conflict detection
- Concerning symptom escalation
- Safe wellness recommendation validation
"""

import pytest
from safety.validator import (
    validate_recommendations,
    check_diagnostic_claims,
    check_medication_claims,
    check_allergy_conflicts,
    check_concerning_symptoms,
    ESCALATION_NOTICE
)


def test_diagnostic_language_detection():
    """Verify that clinical diagnostic claims are identified and rejected."""
    bad_text = "Based on your symptoms, you have diabetes and heart disease."
    issues = check_diagnostic_claims(bad_text)
    assert len(issues) > 0, "Expected diagnostic language to be detected"
    assert any("diabetes" in issue or "diagnostic" in issue.lower() for issue in issues)

    res = validate_recommendations(
        user_input_text="",
        exercise_rec="Start walking.",
        nutrition_rec=bad_text,
        wellness_rec="Sleep 8 hours.",
        user_allergies=[]
    )
    assert res["status"] == "unsafe"
    assert res["action"] == "modify_or_escalate"
    assert len(res["issues"]) > 0


def test_medication_language_detection():
    """Verify that pharmaceutical prescription/dosage advice is caught."""
    bad_text = "Take ibuprofen 400mg twice daily and stop taking your medication if symptoms persist."
    issues = check_medication_claims(bad_text)
    assert len(issues) > 0, "Expected medication/prescription claims to be detected"

    res = validate_recommendations(
        user_input_text="",
        exercise_rec="Light stretching.",
        nutrition_rec="Eat healthy.",
        wellness_rec=bad_text,
        user_allergies=[]
    )
    assert res["status"] == "unsafe"
    assert res["action"] == "modify_or_escalate"


def test_allergy_conflict_detection():
    """Verify that foods conflicting with user's declared allergies are flagged."""
    allergies = ["peanuts", "dairy"]
    conflicting_meal = "For breakfast, enjoy a whole wheat toast with peanut butter and a glass of dairy milk."
    
    issues = check_allergy_conflicts(conflicting_meal, allergies)
    assert len(issues) >= 1, "Expected allergy conflict to be detected"

    res = validate_recommendations(
        user_input_text="",
        exercise_rec="Walk 30 mins.",
        nutrition_rec=conflicting_meal,
        wellness_rec="Hydrate well.",
        user_allergies=allergies
    )
    assert res["status"] == "unsafe"
    assert res["action"] == "modify_or_escalate"


def test_concerning_symptom_escalation():
    """Verify that acute emergency symptoms in user input trigger clinical escalation."""
    emergency_input = "I have chest pain. What disease do I have and what medicine should I take?"
    
    symptom_issues = check_concerning_symptoms(emergency_input)
    assert len(symptom_issues) > 0, "Expected chest pain to be flagged as a concerning symptom"

    res = validate_recommendations(
        user_input_text=emergency_input,
        exercise_rec="Walk 30 minutes.",
        nutrition_rec="Eat more greens.",
        wellness_rec="Rest properly.",
        user_allergies=[]
    )
    assert res["status"] == "unsafe"
    assert res["action"] == "modify_or_escalate"
    assert res["escalation_message"] == ESCALATION_NOTICE
    assert "professional medical evaluation" in res["sanitized_output"]["wellness"]


def test_safe_wellness_recommendation():
    """Verify that general, gradual wellness guidance without medical claims passes safely."""
    safe_exercise = "Try gradually increasing your daily physical activity with 30 minutes of brisk walking."
    safe_nutrition = "Consider including a variety of colorful vegetables and protein-rich lentils."
    safe_wellness = "Maintaining consistent sleep schedules can support overall circadian wellbeing."
    
    res = validate_recommendations(
        user_input_text="Looking for a healthy routine to stay fit.",
        exercise_rec=safe_exercise,
        nutrition_rec=safe_nutrition,
        wellness_rec=safe_wellness,
        user_allergies=["peanuts"]
    )
    assert res["status"] == "safe"
    assert res["action"] == "show"
    assert len(res["issues"]) == 0
    assert res["escalation_message"] is None
