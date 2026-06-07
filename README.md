# Grundfos Pump Predictive Maintenance (Phase 1: Prototype Model)

This repository contains a Python-based predictive analytics module designed to monitor and forecast industrial machinery health, specifically focusing on **Grundfos Centrifugal Pumps**.

## Project Features
- **Exploratory Data Analysis (EDA):** Uses `Seaborn` to visualize the correlation between bearing temperature and vibration levels.
- **Predictive Analytics:** Implements a `Decision Tree Classifier` to auto-detect pump faults (e.g., Cavitation or Bearing Wear).
- **Real-time Inference:** Predicts machine status based on incoming fresh sensor configurations.

## How to Run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Run the script: `python grundfos_pump_analysis.py`
