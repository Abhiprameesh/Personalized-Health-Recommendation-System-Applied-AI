"""
Domain schemas and Pydantic models for Personalized Health AI system.
"""

from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator


class UserProfile(BaseModel):
    """User-provided profile data collected through Streamlit UI."""
    age: int = Field(..., ge=12, le=110, description="Age in years")
    height_cm: float = Field(..., ge=80.0, le=250.0, description="Height in centimeters")
    weight_kg: float = Field(..., ge=30.0, le=250.0, description="Weight in kilograms")
    fitness_goal: str = Field(..., description="Primary fitness/wellness goal")
    activity_level: str = Field(..., description="Baseline activity level: Low, Moderate, or High")
    diet_preference: str = Field(..., description="Dietary preference, e.g., Vegetarian, Vegan, Omnivore")
    exercise_preference: str = Field(..., description="Preferred form of exercise, e.g., Walking, Cycling")
    available_time_mins: int = Field(..., ge=5, le=240, description="Available workout time per day in minutes")
    allergies: List[str] = Field(default_factory=list, description="Declared food allergies and intolerances")
    optional_health_info: Optional[str] = Field(default="", description="Optional self-reported wellness context or symptoms")
    consent_given: bool = Field(default=False, description="Explicit user consent flag")

    @property
    def bmi(self) -> float:
        """Calculate Body Mass Index (BMI)."""
        height_m = self.height_cm / 100.0
        return round(self.weight_kg / (height_m ** 2), 1)

    @property
    def bmi_category(self) -> str:
        """Return qualitative WHO BMI categorization."""
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25.0:
            return "Normal weight"
        elif bmi < 30.0:
            return "Overweight"
        else:
            return "Obesity range"


class ActivityData(BaseModel):
    """Synthetic or tracked activity metrics."""
    user_id: str = Field(..., description="User identifier (e.g. U001)")
    age: int = Field(..., ge=10, le=110)
    steps_avg: int = Field(..., ge=0, description="Average daily steps")
    workout_minutes_avg: int = Field(..., ge=0, description="Average daily workout minutes")
    sleep_hours_avg: float = Field(..., ge=0.0, le=24.0, description="Average daily sleep hours")
    resting_heart_rate: int = Field(..., ge=35, le=160, description="Resting heart rate in beats per minute")
    activity_level: str = Field(..., description="Baseline activity classification: Low, Moderate, High")
    workouts_per_week: int = Field(..., ge=0, le=14, description="Weekly exercise frequency")


class GuidelinePassage(BaseModel):
    """RAG retrieved guideline chunk with authoritative metadata."""
    content: str
    source: str
    topic: str
    title: str
    relevance_score: Optional[float] = None


class SafetyCheckResult(BaseModel):
    """Structured safety validation outcome."""
    status: str = Field(..., description="'safe' or 'unsafe'")
    issues: List[str] = Field(default_factory=list, description="List of detected safety issues")
    action: str = Field(..., description="'show' or 'modify_or_escalate'")
    escalation_message: Optional[str] = None
    sanitized_output: Optional[Dict[str, Any]] = None


class WorkflowState(TypedDict, total=False):
    """LangGraph state passed across all workflow nodes."""
    user_profile: Dict[str, Any]
    activity_data: Dict[str, Any]
    retrieved_context: List[Dict[str, Any]]
    exercise_recommendation: Dict[str, Any]
    nutrition_recommendation: Dict[str, Any]
    wellness_recommendation: Dict[str, Any]
    safety_result: Dict[str, Any]
    final_response: Dict[str, Any]
    consent_valid: bool
    error: Optional[str]
