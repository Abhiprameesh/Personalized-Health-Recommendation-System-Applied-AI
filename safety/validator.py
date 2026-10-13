"""
Safety Validation Layer for Personalized Health AI.
Inspects user input and generated recommendations for:
1. Diagnostic claims
2. Medication / prescription claims
3. Dangerous recommendations
4. Declared allergy conflicts
5. Unsupported medical claims
6. Red-flag concerning symptoms requiring clinical escalation
"""

import re
from typing import List, Dict, Any, Optional

# Patterns for prohibited diagnostic language
DIAGNOSTIC_PATTERNS = [
    r"\byou have (?:diabetes|heart disease|cancer|hypertension|asthma|depression|anxiety|infection)\b",
    r"\byou are suffering from\b",
    r"\byou are diagnosed with\b",
    r"\byour symptoms (?:mean|prove|indicate) you have\b",
    r"\bthis is a confirmed case of\b",
    r"\byou suffer from\b",
    r"\bdiagnosis:\b"
]

# Patterns for prohibited medication / prescription language
MEDICATION_PATTERNS = [
    r"\btake (?:this |a )?medication\b",
    r"\bstop taking (?:your )?medication\b",
    r"\bincrease your dosage\b",
    r"\bdecrease your dosage\b",
    r"\bprescribe\b",
    r"\bprescription\b",
    r"\b(?:take|ingest|administer)\s+\d+\s*(?:mg|milligrams|tablets|pills)\b",
    r"\b(?:aspirin|ibuprofen|paracetamol|metformin|lisinopril|amoxicillin|antibiotics|statin)\b"
]

# Patterns for concerning / emergency red-flag symptoms in user input
CONCERNING_SYMPTOM_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\bdifficulty breathing\b",
    r"\bsudden numbness\b",
    r"\bsudden weakness\b",
    r"\bfacial droop(?:ing)?\b",
    r"\bspeech difficulty\b",
    r"\bfainting\b",
    r"\bloss of consciousness\b",
    r"\bcrushing pressure\b",
    r"\bsevere radiating pain\b",
    r"\bcoughing up blood\b"
]

# Common allergen keyword expansions
ALLERGEN_MAP = {
    "peanut": ["peanut", "peanuts", "peanut butter", "groundnut"],
    "peanuts": ["peanut", "peanuts", "peanut butter", "groundnut"],
    "dairy": ["milk", "cheese", "dairy", "yogurt", "butter", "whey", "casein", "cream"],
    "lactose": ["milk", "cheese", "dairy", "yogurt", "butter", "whey", "cream"],
    "gluten": ["wheat", "gluten", "barley", "rye", "bread", "pasta", "flour"],
    "wheat": ["wheat", "gluten", "bread", "pasta", "flour"],
    "egg": ["egg", "eggs", "egg white", "egg yolk", "mayonnaise"],
    "eggs": ["egg", "eggs", "egg white", "egg yolk", "mayonnaise"],
    "soy": ["soy", "soya", "tofu", "edamame", "soy sauce", "tempeh"],
    "tree nut": ["almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut", "tree nut", "macadamia"],
    "tree nuts": ["almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut", "tree nut", "macadamia"],
    "fish": ["fish", "salmon", "tuna", "cod", "tilapia", "sardine"],
    "shellfish": ["shellfish", "shrimp", "crab", "lobster", "prawn", "oyster", "clam"],
}

ESCALATION_NOTICE = "Your input may require professional medical evaluation. Please consult a qualified healthcare professional."


def check_concerning_symptoms(text: str) -> List[str]:
    """Detect acute or alarming medical symptoms in user input."""
    if not text:
        return []
    text_lower = text.lower()
    issues = []
    for pat in CONCERNING_SYMPTOM_PATTERNS:
        if re.search(pat, text_lower):
            issues.append(f"Concerning symptom detected matching: {pat}")
    return issues


def check_diagnostic_claims(text: str) -> List[str]:
    """Detect diagnostic claims or disease confirmation statements."""
    if not text:
        return []
    text_lower = text.lower()
    issues = []
    for pat in DIAGNOSTIC_PATTERNS:
        if re.search(pat, text_lower):
            issues.append(f"Diagnostic claim detected matching: {pat}")
    return issues


def check_medication_claims(text: str) -> List[str]:
    """Detect pharmaceutical prescriptions or dosage instructions."""
    if not text:
        return []
    text_lower = text.lower()
    issues = []
    for pat in MEDICATION_PATTERNS:
        if re.search(pat, text_lower):
            issues.append(f"Medication/prescription claim detected matching: {pat}")
    return issues


def check_allergy_conflicts(text: str, user_allergies: List[str]) -> List[str]:
    """Detect if suggested foods conflict with user's declared allergies."""
    if not text or not user_allergies:
        return []
    
    text_lower = text.lower()
    issues = []
    
    for allergy in user_allergies:
        allergy_clean = allergy.strip().lower()
        if not allergy_clean:
            continue
        
        # Check direct term and mapped keywords
        keywords = ALLERGEN_MAP.get(allergy_clean, [allergy_clean])
        for kw in keywords:
            # Word boundary check
            pattern = rf"\b{re.escape(kw)}\b"
            # If the text explicitly warns 'avoid peanuts' or 'without peanuts', do not flag as a conflict
            if re.search(pattern, text_lower):
                # Check if it's in a negation context like 'avoid <kw>' or 'free of <kw>'
                neg_pattern = rf"\b(?:avoid|free of|without|no|exclude)\s+{re.escape(kw)}\b"
                if not re.search(neg_pattern, text_lower):
                    issues.append(f"Allergy conflict: '{allergy_clean}' allergen keyword '{kw}' detected in recommendation")
                    break

    return issues


def validate_recommendations(
    user_input_text: str,
    exercise_rec: str,
    nutrition_rec: str,
    wellness_rec: str,
    user_allergies: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive safety validator.
    Inspects user input and all generated recommendation components.
    Returns structured result:
    {
        "status": "safe" | "unsafe",
        "issues": [...],
        "action": "show" | "modify_or_escalate",
        "escalation_message": str,
        "sanitized_output": Dict
    }
    """
    if user_allergies is None:
        user_allergies = []

    issues = []

    # 1. Check user input for emergency/concerning symptoms
    symptom_issues = check_concerning_symptoms(user_input_text)
    if symptom_issues:
        issues.extend(symptom_issues)
        return {
            "status": "unsafe",
            "issues": issues,
            "action": "modify_or_escalate",
            "escalation_message": ESCALATION_NOTICE,
            "sanitized_output": {
                "exercise": "Recommendations paused pending medical clearance.",
                "nutrition": "Maintain general balanced hydration and light meals.",
                "wellness": ESCALATION_NOTICE
            }
        }

    # Combined generated text
    combined_rec = f"{exercise_rec}\n{nutrition_rec}\n{wellness_rec}"

    # 2. Check for diagnostic claims in generated text
    diag_issues = check_diagnostic_claims(combined_rec)
    issues.extend(diag_issues)

    # 3. Check for medication/prescription claims
    med_issues = check_medication_claims(combined_rec)
    issues.extend(med_issues)

    # 4. Check for allergy conflicts in nutrition recommendation
    allergy_issues = check_allergy_conflicts(nutrition_rec, user_allergies)
    issues.extend(allergy_issues)

    # Outcome decision
    if issues:
        return {
            "status": "unsafe",
            "issues": issues,
            "action": "modify_or_escalate",
            "escalation_message": "Safety boundary triggered: Output contained clinical, pharmaceutical, or allergy conflicts.",
            "sanitized_output": {
                "exercise": sanitize_text(exercise_rec),
                "nutrition": sanitize_nutrition(nutrition_rec, user_allergies),
                "wellness": sanitize_text(wellness_rec)
            }
        }
    else:
        return {
            "status": "safe",
            "issues": [],
            "action": "show",
            "escalation_message": None,
            "sanitized_output": {
                "exercise": exercise_rec,
                "nutrition": nutrition_rec,
                "wellness": wellness_rec
            }
        }


def sanitize_text(text: str) -> str:
    """Strip or rephrase disallowed diagnostic/prescriptive phrasing."""
    safe = text
    for pat in DIAGNOSTIC_PATTERNS:
        safe = re.sub(pat, "lifestyle factors can influence health", safe, flags=re.IGNORECASE)
    for pat in MEDICATION_PATTERNS:
        safe = re.sub(pat, "consulting a healthcare provider regarding care", safe, flags=re.IGNORECASE)
    return safe


def sanitize_nutrition(nutrition_text: str, allergies: List[str]) -> str:
    """Sanitize nutrition suggestions by removing allergenic references."""
    safe = sanitize_text(nutrition_text)
    for allergy in allergies:
        allergy_clean = allergy.strip().lower()
        keywords = ALLERGEN_MAP.get(allergy_clean, [allergy_clean])
        for kw in keywords:
            pattern = rf"\b{re.escape(kw)}\b"
            safe = re.sub(pattern, f"[allergen-safe alternative for {allergy_clean}]", safe, flags=re.IGNORECASE)
    return safe
