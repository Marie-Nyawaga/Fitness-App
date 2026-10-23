# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# ===========================================
# Load data and train models (runs once)
# ===========================================
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("Fitness.csv")

    # Preprocessing
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df.drop_duplicates(inplace=True)

    # Features and target
    features = [
        'TotalDistance', 'TrackerDistance', 'VeryActiveDistance',
        'ModeratelyActiveDistance', 'LightActiveDistance', 'SedentaryActiveDistance',
        'VeryActiveMinutes', 'FairlyActiveMinutes',
        'LightlyActiveMinutes', 'SedentaryMinutes', 'TotalSteps'
    ]
    X = df[features]
    y = df['Calories']

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    for model in models.values():
        model.fit(X_scaled, y)

    return models, scaler, features, df

# Load once
models, scaler, features, df = load_and_train()

# ===========================================
# Prediction function
# ===========================================
def predict_exact_calories(
    total_distance, tracker_distance, very_active_distance,
    moderately_active_distance, light_active_distance, sedentary_active_distance,
    very_active_minutes, fairly_active_minutes,
    lightly_active_minutes, sedentary_minutes, total_steps
):
    input_data = np.array([[
        total_distance, tracker_distance, very_active_distance,
        moderately_active_distance, light_active_distance, sedentary_active_distance,
        very_active_minutes, fairly_active_minutes,
        lightly_active_minutes, sedentary_minutes, total_steps
    ]])
    input_scaled = scaler.transform(input_data)
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        pred = max(0, pred)  # No negative calories
        predictions[name] = pred
    avg = np.mean(list(predictions.values()))
    return predictions, avg

# ===========================================
# Helper: Get safe min/max from data
# ===========================================
def get_safe_bounds(col, df, margin=0.1, min_floor=0):
    if col not in df.columns:
        return min_floor, 10000
    col_min = df[col].min()
    col_max = df[col].max()
    lower = max(min_floor, col_min * (1 - margin))
    upper = col_max * (1 + margin)
    return float(lower), float(upper)

# ===========================================
# Streamlit UI
# ===========================================
def main():
    st.set_page_config(page_title="Calories Predictor", layout="centered")
    
    # =============================
    # 🎯 Introductory Template
    # =============================
    st.title("🔥 Fitness Calories Burned Predictor")
    
    st.markdown("""
    Welcome! This app uses **machine learning models trained on real fitness tracker data** to estimate how many calories you burned based on your daily activity.

    ### 📥 How to Use:
    - Fill in your activity metrics in the form below.
    - Values **update automatically** — no need to click a button!
    - All inputs are validated to stay within realistic ranges.

    ### ℹ️ What Do These Fields Mean?
    - **Total Distance**: Total km you traveled (walking, running, etc.).
    - **Very Active Minutes**: Time spent in intense activity (e.g., running).
    - **Sedentary Minutes**: Time spent sitting or lying down.
    - **Total Steps**: Total number of steps taken in the day.

    > 💡 **Tip**: If you use a Fitbit, Apple Watch, or Google Fit — these numbers are often available in your daily report!

    ---
    """)

    # =============================
    # 📊 Input Form (unchanged logic, just below intro)
    # =============================
    # --- Distance inputs (floats) ---
    total_dist_min, total_dist_max = get_safe_bounds('TotalDistance', df)
    tracker_dist_min, tracker_dist_max = get_safe_bounds('TrackerDistance', df)
    very_dist_min, very_dist_max = get_safe_bounds('VeryActiveDistance', df)
    mod_dist_min, mod_dist_max = get_safe_bounds('ModeratelyActiveDistance', df)
    light_dist_min, light_dist_max = get_safe_bounds('LightActiveDistance', df)
    sed_dist_min, sed_dist_max = get_safe_bounds('SedentaryActiveDistance', df)

    total_distance = st.number_input("Total Distance (km)", min_value=total_dist_min, max_value=total_dist_max, value=min(5.0, total_dist_max), step=0.1, format="%.2f")
    tracker_distance = st.number_input("Tracker Distance (km)", min_value=tracker_dist_min, max_value=tracker_dist_max, value=min(5.0, tracker_dist_max), step=0.1, format="%.2f")
    very_active_distance = st.number_input("Very Active Distance (km)", min_value=very_dist_min, max_value=very_dist_max, value=min(2.0, very_dist_max), step=0.1, format="%.2f")
    moderately_active_distance = st.number_input("Moderately Active Distance (km)", min_value=mod_dist_min, max_value=mod_dist_max, value=min(1.0, mod_dist_max), step=0.1, format="%.2f")
    light_active_distance = st.number_input("Light Active Distance (km)", min_value=light_dist_min, max_value=light_dist_max, value=min(2.0, light_dist_max), step=0.1, format="%.2f")
    sedentary_active_distance = st.number_input("Sedentary Active Distance (km)", min_value=sed_dist_min, max_value=sed_dist_max, value=0.0, step=0.1, format="%.2f")

    # --- Time & Steps (integers) ---
    very_min_min, very_min_max = get_safe_bounds('VeryActiveMinutes', df)
    fairly_min_min, fairly_min_max = get_safe_bounds('FairlyActiveMinutes', df)
    lightly_min_min, lightly_min_max = get_safe_bounds('LightlyActiveMinutes', df)
    sedentary_min_min, sedentary_min_max = get_safe_bounds('SedentaryMinutes', df)
    steps_min, steps_max = get_safe_bounds('TotalSteps', df)

    very_active_minutes = st.number_input("Very Active Minutes", min_value=int(very_min_min), max_value=int(very_min_max), value=min(30, int(very_min_max)), step=1)
    fairly_active_minutes = st.number_input("Fairly Active Minutes", min_value=int(fairly_min_min), max_value=int(fairly_min_max), value=min(20, int(fairly_min_max)), step=1)
    lightly_active_minutes = st.number_input("Lightly Active Minutes", min_value=int(lightly_min_min), max_value=int(lightly_min_max), value=min(100, int(lightly_min_max)), step=1)
    sedentary_minutes = st.number_input("Sedentary Minutes", min_value=int(sedentary_min_min), max_value=int(sedentary_min_max), value=min(600, int(sedentary_min_max)), step=1)
    total_steps = st.number_input("Total Steps", min_value=int(steps_min), max_value=int(steps_max), value=min(8000, int(steps_max)), step=100)

    # --- Auto-prediction ---
    predictions, avg_pred = predict_exact_calories(
        total_distance, tracker_distance, very_active_distance,
        moderately_active_distance, light_active_distance, sedentary_active_distance,
        very_active_minutes, fairly_active_minutes,
        lightly_active_minutes, sedentary_minutes, total_steps
    )

    # --- Results ---
    st.divider()
    st.subheader("📊 Prediction Results")
    for model, calories in predictions.items():
        st.write(f"**{model}**: {calories:.1f} kcal")

    st.success(f"**Average Estimated Calories Burned**: {avg_pred:.1f} kcal")

    st.subheader("🎯 Activity Level")
    if avg_pred < 1500:
        st.warning("Low Activity: Fewer calories than average.")
    elif avg_pred < 2000:
        st.info("Moderate Activity: Balanced movement today!")
    else:
        st.success("High Activity: Excellent effort! 💪")

# ===========================================
# Run app
# ===========================================
if __name__ == "__main__":
    main()