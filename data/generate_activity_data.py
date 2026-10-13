"""
Script to generate synthetic activity data for 300 demonstration users.
This data is purely synthetic and does not represent real individuals or clinical patients.
"""

import csv
import random
from pathlib import Path

def generate_dataset(output_path: Path, num_users: int = 300):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    random.seed(42)  # For reproducible synthetic data
    
    rows = []
    
    # Pre-seed User A and User B exactly as defined in Case Study 19 presentation
    rows.append({
        "user_id": "U001",
        "age": 21,
        "steps_avg": 4200,
        "workout_minutes_avg": 15,
        "sleep_hours_avg": 6.2,
        "resting_heart_rate": 78,
        "activity_level": "Low",
        "workouts_per_week": 1
    })
    
    rows.append({
        "user_id": "U002",
        "age": 45,
        "steps_avg": 8500,
        "workout_minutes_avg": 40,
        "sleep_hours_avg": 7.4,
        "resting_heart_rate": 71,
        "activity_level": "Moderate",
        "workouts_per_week": 4
    })
    
    rows.append({
        "user_id": "U003",
        "age": 34,
        "steps_avg": 11000,
        "workout_minutes_avg": 60,
        "sleep_hours_avg": 8.0,
        "resting_heart_rate": 66,
        "activity_level": "High",
        "workouts_per_week": 5
    })
    
    activity_categories = ["Low", "Moderate", "High"]
    
    for i in range(4, num_users + 1):
        uid = f"U{i:03d}"
        level = random.choices(activity_categories, weights=[0.35, 0.45, 0.20])[0]
        age = random.randint(19, 72)
        
        if level == "Low":
            steps = random.randint(2500, 5500)
            workout_mins = random.randint(5, 25)
            sleep = round(random.uniform(5.5, 7.2), 1)
            rhr = random.randint(72, 86)
            workouts = random.randint(0, 2)
        elif level == "Moderate":
            steps = random.randint(6000, 9500)
            workout_mins = random.randint(30, 50)
            sleep = round(random.uniform(6.5, 8.0), 1)
            rhr = random.randint(65, 76)
            workouts = random.randint(3, 4)
        else:  # High
            steps = random.randint(10000, 15500)
            workout_mins = random.randint(50, 90)
            sleep = round(random.uniform(7.0, 8.8), 1)
            rhr = random.randint(56, 68)
            workouts = random.randint(5, 6)
            
        rows.append({
            "user_id": uid,
            "age": age,
            "steps_avg": steps,
            "workout_minutes_avg": workout_mins,
            "sleep_hours_avg": sleep,
            "resting_heart_rate": rhr,
            "activity_level": level,
            "workouts_per_week": workouts
        })
        
    fieldnames = [
        "user_id", "age", "steps_avg", "workout_minutes_avg", 
        "sleep_hours_avg", "resting_heart_rate", "activity_level", "workouts_per_week"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} synthetic user activity records at {output_path}")

if __name__ == "__main__":
    csv_file = Path(__file__).resolve().parent / "activity_data.csv"
    generate_dataset(csv_file, 300)
