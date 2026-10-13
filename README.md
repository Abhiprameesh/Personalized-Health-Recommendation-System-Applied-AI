# Personalized Health Recommendation System
### AI-Based Design for Preventive Wellness
**Applied AI Case Study 19 (22AI741)**  
*Author:* Abhinand Prameesh (USN: 1DS23AI002)

---

## 1. Project Objective
The objective of this project is to develop a working proof-of-concept AI recommendation system for **preventive wellness**. The system generates personalized physical activity, nutrition, and daily recovery guidance by combining user-provided preferences with synthetic wearable activity patterns, grounding recommendations in authoritative public health guidelines via Retrieval-Augmented Generation (RAG), and enforcing clinical safety boundaries.

> **Important Educational Note:** This is an Applied AI case study and educational prototype designed for demonstration and academic defense. It is **not** a clinical diagnostic tool or medical treatment system.

---

## 2. Problem Statement
Generic wellness guidance fails because it provides one-size-fits-all advice (e.g., standard generic diets or fixed exercise routines) that ignores individual baseline fitness, schedule constraints, food intolerances, and recovery markers. This leads to low adherence and potential injury. Conversely, a personalized AI approach adapts dynamically to user goals, baseline activity data, and dietary preferences while strictly operating within preventive wellness boundaries and respecting user privacy.

---

## 3. System Architecture
The system realizes the 5-step pipeline and architectural design from the Case Study 19 presentation:

```
                      +-----------------------------+
                      |            USER             |
                      | (Profile, Health, Activity) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   CONSENT & PRIVACY LAYER   |
                      |  (Opt-in Check, Zero Leaks) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    DATA PROCESSING LAYER    |
                      |  (Pydantic Models + Pandas) |
                      +--------------+--------------+
                                     |
                                     v
                 +---------------------------------------+
                 |       AI RECOMMENDATION ENGINE        |
                 |         (LangGraph Workflow)          |
                 |                                       |
                 |  +---------------+ +---------------+  |
                 |  |Exercise Agent | |Nutrition Agent|  |
                 |  +---------------+ +---------------+  |
                 |          +-------------------+        |
                 |          |  Wellness Agent   |        |
                 |          +-------------------+        |
                 +-------------------+-------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    RAG KNOWLEDGE GROUNDING  |
                      |  (ChromaDB + WHO/CDC/AHA)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  SAFETY & VALIDATION LAYER  |
                      | (Rule Checks, Allergen Scan,|
                      |    Clinical Red-Flags)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     PERSONALIZED OUTPUT     |
                      |  (Streamlit UI, Citations,  |
                      |     Safety Audit Logs)      |
                      +-----------------------------+
```

---

## 4. Data Strategy
The system ingests three distinct, privacy-preserving data sources:
1. **User Profile (Streamlit Form):** Age, Height, Weight, Fitness Goal, Activity Level, Diet Preference, Exercise Preference, Available Time, Declared Allergies, and Optional Self-Reported Health Info (validated via Pydantic `UserProfile`).
2. **Synthetic Activity Metrics (`data/activity_data.csv`):** Pre-generated realistic activity profiles for 300 synthetic users (`U001` - `U300`) with step counts, workout minutes, sleep duration, resting heart rate, and workouts per week.
3. **Authoritative Knowledge Base (`knowledge/*.md`):** Curated public health documents with structured metadata (`source`, `topic`, `title`).

---

## 5. Why Synthetic Data is Used
- **Patient Privacy:** Real patient datasets contain Protected Health Information (PHI/PII) governed by HIPAA, GDPR, and clinical ethics regulations.
- **Educational Scope:** An applied AI demonstration does not require training deep clinical models; synthetic data provides realistic variance without privacy risks or ethical liability.
- **Reproducibility:** Seeded synthetic users (User A `U001` and User B `U002`) ensure verifiable, deterministic evaluation across different lifestyle cohorts.

---

## 6. How LangGraph Works
The workflow is implemented using **LangGraph** (`workflow/graph.py`), maintaining a stateful `WorkflowState` dictionary passed across 8 dedicated nodes:
1. `build_profile`: Validates inputs and enforces the consent gate.
2. `load_activity_data`: Fetches synthetic records from CSV or accepts manual metrics.
3. `retrieve_guidelines`: Queries ChromaDB for top-ranked public health guidelines.
4. `exercise_agent`: Formulates gradual aerobic and progressive training plans.
5. `nutrition_agent`: Formulates balanced whole-food meal templates.
6. `wellness_agent`: Analyzes sleep and resting heart rate recovery.
7. `safety_validator`: Audits recommendations against medical and allergen constraints.
8. `final_formatter`: Formats output cards, appends citations, and attaches disclaimers.

---

## 7. The Three Specialized Agents
- **Exercise Agent (`agents/exercise_agent.py`):** Formulates realistic plans based on age, fitness goal, baseline activity, and available time. Recommends progressive overload over 2-4 weeks (e.g. brisk walking for low-activity User A; structured cycling intervals for moderate User B). Never prescribes extreme routines.
- **Nutrition Agent (`agents/nutrition_agent.py`):** Adapts dietary suggestions to the user's preference (Vegetarian, Vegan, Omnivore) and macro goals. **Strictly excludes all declared allergens** (peanuts, dairy, gluten, shellfish, etc.). Never claims to cure or manage medical conditions.
- **Preventive Wellness Agent (`agents/wellness_agent.py`):** Evaluates sleep duration (flagging <7h baselines for circadian nudges) and resting heart rate (>75 bpm prompts parasympathetic breathwork). Immediately routes acute symptoms to clinical escalation.

---

## 8. How RAG Works
- **Storage:** Local **ChromaDB** vector database (`chroma_db/`) populated by `rag/ingest.py` from 5 markdown files in `knowledge/`.
- **Retrieval:** `rag/retriever.py` queries ChromaDB with metadata filtering across `physical activity`, `nutrition`, `sleep wellness`, and `preventive wellness`.
- **Transparency:** The UI provides an expander showing the exact guideline passages, source organizations (WHO, CDC, AHA, Sleep Foundation), and document titles.
- **Hallucination Caveat:** The documentation and UI explicitly state:
  > *"RAG helps reduce unsupported claims by grounding recommendations in retrieved knowledge, but it does not guarantee factual correctness."*

---

## 9. How Safety Validation Works
The Safety Layer (`safety/validator.py`) inspects user input and generated plans for:
1. **Diagnostic Claims:** Blocks phrases like *"You have diabetes"*, *"Your symptoms prove you have..."*.
2. **Medication & Prescriptions:** Blocks pharmaceutical advice, drug names (*"aspirin"*, *"ibuprofen"*), and dosage instructions.
3. **Allergy Conflicts:** Validates that no declared allergen appears in food suggestions.
4. **Emergency Red Flags:** Detects acute symptoms in user input (e.g. *"chest pain"*, *"shortness of breath"*, *"sudden numbness"*) and immediately halts recommendation generation, presenting the clinical escalation alert:
   > *"Your input may require professional medical evaluation. Please consult a qualified healthcare professional."*

Returns structured output:
```json
{
  "status": "safe",
  "issues": [],
  "action": "show"
}
```
or:
```json
{
  "status": "unsafe",
  "issues": ["Concerning symptom detected matching: chest pain"],
  "action": "modify_or_escalate",
  "escalation_message": "Your input may require professional medical evaluation..."
}
```

---

## 10. How Consent Works
- Before processing any profile or activity data, the user must explicitly toggle the opt-in checkbox:
  > *"I consent to using the information I provide to generate personalized wellness recommendations."*
- If unchecked, the LangGraph `router_consent_check` immediately short-circuits the pipeline to `END` and returns a clear refusal notice without processing personal data.

---

## 11. Installation
1. Clone the repository and navigate to the project root:
   ```bash
   cd c:\Personalize-Health-AppliedAI
   ```
2. Activate the dedicated virtual environment:
   ```powershell
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```
3. (Optional) Re-install requirements if setting up fresh:
   ```bash
   pip install -r requirements.txt
   ```

---

## 12. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```
> **Offline / Resilient Mode:** An API key is **optional**. If omitted, the system runs its high-quality, deterministic grounded rule-engine using RAG guidelines and user parameters. When a Gemini API key is added, the agents automatically invoke Google Gemini.

---

## 13. How to Run
### Start the Streamlit Application:
```powershell
.venv\Scripts\python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run Automated Unit Tests:
```powershell
.venv\Scripts\python -m pytest tests/ -v
```

---

## 14. Demonstration Scenarios (Viva Checklist)
| Demo | Description | Expected System Behavior |
|---|---|---|
| **Demo 1** | Load Example User A (Age 21, Low activity, Walking, Veg) | Walking progression plan (5x/week, 30 min), plant protein/iron tips, sleep hygiene. Safety check: **PASSED**. |
| **Demo 2** | Load Example User B (Age 45, Moderate activity, Cycling, Veg) | Moderate cycling routine (40-60 min, 3-4x/week), recovery intervals, balanced macros. Safety check: **PASSED**. |
| **Demo 3** | Change user's goal/exercise preference (e.g. Swimming, Muscle tone) | Output updates immediately to reflect new exercise and nutrition priorities. |
| **Demo 4** | Enter food allergy (e.g. `peanuts` or `dairy`) | Allergen filter activates, conflicting items excluded from meal suggestions, safety check passes. |
| **Demo 5** | Enter clinical symptom: *"I have chest pain. What disease do I have and what medicine should I take?"* | System **refuses** diagnosis/prescription; displays mandatory medical escalation notice. |
| **Demo 6** | Uncheck consent checkbox and click Generate | System immediately halts processing and displays consent rejection alert. |

---

## 15. Limitations
- **Synthetic Activity Data:** Activity metrics are algorithmically simulated and do not capture real-world biosensor noise.
- **Non-Clinical Scope:** Designed exclusively for preventive lifestyle habits; does not treat, diagnose, or manage acute/chronic conditions.
- **RAG Grounding:** Grounding significantly reduces hallucination risk, but does not guarantee 100% factual perfection.
- **Prototype Rules:** Safety checks use rule-based regex patterns suitable for prototypes; enterprise deployment requires medical-grade NLP and clinical human review.

---

## 16. Medical Disclaimer
> *"This prototype provides general wellness information and is not a medical diagnostic or treatment system. RAG helps reduce unsupported claims by grounding recommendations in retrieved knowledge, but it does not guarantee factual correctness. Always consult a qualified healthcare professional before beginning any new exercise or dietary regimen."*
