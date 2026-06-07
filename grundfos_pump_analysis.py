import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Grundfos Pump ka Dummy Data (Jo hum ne analyze karna shuru kiya tha)
pump_data = {
    'Timestamp': pd.date_range(start='2026-06-01', periods=10, freq='h'),
    'Vibration_mm_s': [2.1, 2.3, 2.0, 4.5, 4.8, 5.2, 2.2, 2.1, 5.5, 5.9], # 4.0 se ooper matlab masla he
    'Temperature_C': [45, 46, 44, 58, 62, 65, 45, 46, 68, 72],          # Ooper vibration pr temp bhi barh rhi he
    'Flow_Rate_m3_h': [50, 49, 51, 42, 38, 35, 50, 51, 32, 28],          # Fault pr flow rate kam ho rha he
}

df = pd.DataFrame(pump_data)

# 2. Target Variable banana (Decision Tree/Logistic Regression ke liye Condition set karna)
# Agar vibration 4.0 mm/s se zyada he to 'Fault' (1), warna 'Normal' (0)
df['Pump_Status'] = np.where(df['Vibration_mm_s'] > 4.0, 1, 0)

print("--- Grundfos Pump Data Summary ---")
print(df)

# --- INCOMPLETE PART (Jo hum ne aaj complete karna tha) ---
# 3. Data Visualization (Seaborn ke zariye check karna ke Temperature aur Vibration ka kya rishta he)

plt.figure(figsize=(8, 5))
# Seaborn ka scatter plot use kar rahe hain status ke mutabiq rang badalne ke liye
sns.scatterplot(data=df, x='Temperature_C', y='Vibration_mm_s', hue='Pump_Status', palette='coolwarm', s=100)

plt.title('Grundfos Pump: Temperature vs Vibration Analysis')
plt.xlabel('Bearing Temperature (°C)')
plt.ylabel('Vibration Level (mm/s)')
plt.grid(True)
plt.show()

# --- MACHINE LEARNING MODULE: DECISION TREE ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# 1. Features (X) aur Target (y) alag karna
# Hum Temperature aur Vibration ki bunyad par faisla kar rahe hain
X = df[['Temperature_C', 'Vibration_mm_s']] 
y = df['Pump_Status']

# 2. Decision Tree Model ko Initialize karna
model = DecisionTreeClassifier(max_depth=3)

# 3. Model ko Train karna (Pump ka data parhana)
model.fit(X, y)

print("\n--- Decision Tree Model Training Complete! ---")

# 4. NAYE DATA PAR PREDICTION (Ab hum naya data de kar check karte hain)
# Maan lain plant par naya data aaya: Temp = 60°C, Vibration = 4.6 mm/s
naya_data = [[60, 4.6]]
prediction = model.predict(naya_data)

if prediction[0] == 1:
    print("WARNING: Model Predicts -> PUMP FAULT DETECTED!")
else:
    print("STATUS: Model Predicts -> Pump is Operating Normally.")

# 5. BONUS: Decision Tree ka apna graphical flowchart dekhna
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=['Normal', 'Fault'], filled=True)
plt.title("Grundfos Pump Fault Detection - Decision Tree Logic")
plt.show()
