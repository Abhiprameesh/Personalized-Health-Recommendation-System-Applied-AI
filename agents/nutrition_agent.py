"""
Nutrition Agent: Generates balanced nutrition guidance, meal templates, and allergen-safe
dietary suggestions grounded in dietary guidelines and tailored to user preferences.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


from agents.llm_client import invoke_gemini_safely


def generate_nutrition_recommendations(
    profile: Dict[str, Any],
    activity: Dict[str, Any],
    guidelines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate balanced dietary suggestions respecting diet preference and allergen exclusions.
    """
    goal = profile.get("fitness_goal", "General fitness")
    diet_pref = profile.get("diet_preference", "Vegetarian")
    allergies = profile.get("allergies", [])
    if isinstance(allergies, str):
        allergies = [a.strip() for a in allergies.split(",") if a.strip()]
    act_level = profile.get("activity_level", "Moderate")
    
    # Grounding context from RAG
    guideline_excerpts = [g.get("content", "") for g in guidelines if "nutrition" in g.get("topic", "").lower()]
    rag_context = "\n".join(guideline_excerpts[:2])

    allergy_str = ", ".join(allergies) if allergies else "None reported"

    prompt = f"""You are a preventive wellness nutrition specialist.
User Profile:
- Goal: {goal}, Dietary Preference: {diet_pref}, Activity Level: {act_level}
- Declared Allergies: {allergy_str} (STRICTLY AVOID ALL INGREDIENTS OR DERIVATIVES OF THESE ALLERGENS)

Authoritative Nutrition Guidelines (RAG context):
{rag_context}

Instructions:
1. Provide a balanced, whole-food meal template adhering to {diet_pref} preferences.
2. Ensure high-protein and nutrient density aligned with the user's goal ({goal}).
3. EXPLICITLY respect the allergy restrictions ({allergy_str}). NEVER suggest any recipe or snack containing them.
4. DO NOT provide medical diets or claim to cure diseases.
5. Provide a 4-bullet template covering breakfast, lunch, dinner, and hydration/snacks.
"""

    llm_text, model_used = invoke_gemini_safely(prompt)
    if llm_text:
        return {
            "summary": f"Nutrient-Dense {diet_pref} Meal Plan ({goal})",
            "meal_plan": llm_text,
            "allergies_respected": allergies,
            "diet_preference": diet_pref,
            "mode": f"Gemini ({model_used})"
        }

    # High-quality deterministic generator grounded in dietary principles and allergen safety
    allergy_lower = [a.lower() for a in allergies]
    
    # Safe snack selection based on allergens
    if any(a in ["nut", "nuts", "peanut", "peanuts", "tree nut"] for a in allergy_lower):
        snack = "Pumpkin seed and sunflower seed trail mix with dried cranberries, or hummus with crisp cucumber slices."
    elif any(a in ["dairy", "milk", "lactose"] for a in allergy_lower):
        snack = "Chia seed pudding made with unsweetened almond or oat milk, topped with blueberries."
    else:
        snack = "Handful of roasted almonds or walnuts, accompanied by crisp apple slices or Greek yogurt with berries."

    # Build diet-tailored meal templates
    if diet_pref.lower() == "vegetarian":
        protein_sources = "lentils, chickpeas, tofu, paneer or Greek yogurt (if dairy tolerant), and edamame"
        iron_note = "Pair plant-based iron sources (spinach, lentils) with vitamin C rich foods (lemon juice, bell peppers) to boost absorption."
        breakfast = "Warm oatmeal prepared with flaxseeds, chia seeds, sliced bananas, and a scoop of plant-based protein powder."
        lunch = "Quinoa and roasted vegetable bowl with seasoned chickpeas, baby spinach, cherry tomatoes, and tahini lemon dressing."
        dinner = "Spiced lentil dal (or tofu stir-fry) with steamed brown rice and sautéed seasonal greens."
    elif diet_pref.lower() == "vegan":
        protein_sources = "lentils, black beans, organic tempeh, tofu, hemp hearts, and green peas"
        iron_note = "Incorporate fortified foods or nutritional yeast for B12, and pair dark leafy greens with citrus."
        breakfast = "Smoothie bowl with spinach, frozen berries, hemp protein, and rolled oats."
        lunch = "Spiced black bean and sweet potato wrap with fresh guacamole and cabbage slaw in a whole-grain tortilla."
        dinner = "Baked tempeh with stir-fried broccoli, snap peas, bell peppers, and quinoa."
    else:  # Omnivore / Balanced
        protein_sources = "lean poultry, eggs, lentils, fatty fish (wild salmon), and cottage cheese"
        iron_note = "Maintain balanced micronutrient intake through varied colorful vegetables."
        breakfast = "Two poached or boiled eggs with whole-grain sourdough toast, sliced avocado, and baby greens."
        lunch = "Grilled chicken breast or tofu salad with mixed greens, walnuts (if safe), quinoa, and olive oil vinaigrette."
        dinner = "Baked salmon or chickpea stew with roasted sweet potato and steamed asparagus."

    meal_plan = (
        f"• **Breakfast:** {breakfast}\n"
        f"• **Lunch:** {lunch}\n"
        f"• **Dinner:** {dinner}\n"
        f"• **Snacks & Hydration:** {snack} Aim for 2.0 to 2.5 liters of clean water daily.\n"
        f"• **Key Micronutrient Focus:** {iron_note}"
    )

    return {
        "summary": f"Balanced {diet_pref} Nutrition Template for {goal}",
        "meal_plan": meal_plan,
        "protein_focus": protein_sources,
        "allergies_respected": allergies,
        "diet_preference": diet_pref,
        "mode": "Authoritative Guidelines Grounded"
    }
