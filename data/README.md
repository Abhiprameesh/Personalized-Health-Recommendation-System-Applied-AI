# Synthetic Activity Dataset

## Demonstration Disclaimer
The data in `activity_data.csv` is **completely synthetic** and was generated algorithmically for educational and demonstration purposes as part of the **Personalized Health Recommendation System (Applied AI Case Study 19)**.

- **No Real Patients:** This dataset does not contain records from real individuals, patients, clinical studies, or health records.
- **No Personal Identifiable Information (PII):** All user identifiers (`U001` - `U300`) and associated metrics are artificially created.
- **Purpose:** The dataset is used solely to demonstrate activity-data integration, personalization logic, and profile aggregation within the LangGraph multi-agent recommendation engine. No deep learning or diagnostic clinical model is trained on this data.

## Schema
- `user_id`: Synthetic identifier (e.g. `U001`, `U002`)
- `age`: Simulated age in years
- `steps_avg`: Daily average step count
- `workout_minutes_avg`: Average duration of workouts in minutes
- `sleep_hours_avg`: Average daily sleep duration in hours
- `resting_heart_rate`: Estimated resting heart rate (bpm)
- `activity_level`: Categorical baseline (`Low`, `Moderate`, `High`)
- `workouts_per_week`: Frequency of weekly exercise sessions
