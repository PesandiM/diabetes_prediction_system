import pandas as pd

def preprocess_input(data_dict):
    mappings = {
        'gender': {'Female': 1, 'Male': 2, 'Other': 3},
        'ethnicity': {'Hispanic': 1, 'White': 2, 'Asian': 3, 'Black': 4, 'Other': 5},
        'income_level': {'Lower-Middle': 1, 'Upper-Middle': 2, 'Low': 3, 'Middle': 4, 'High': 5},
        'education_level': {'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3, 'No formal': 4},
        'smoking_status': {'Current': 1, 'Never': 2, 'Former': 3},
        'employment_status': {'Employed': 1, 'Retired': 2, 'Student': 3, 'Unemployed': 4}
    }

    # 2. Create DataFrame
    df = pd.DataFrame([data_dict])

    # 3. Apply Mappings
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    column_order = [
        'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
        'gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status',
        'employment_status', 'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history'
    ]

    return df[column_order]