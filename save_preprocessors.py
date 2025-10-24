"""
Script to save preprocessing components (scaler and feature selector) from training
Run this after training the model to save the preprocessing pipeline
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR

# Load data
dir_path = './data/CMaps/'
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names

train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)

# Add RUL
def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

train = add_remaining_useful_life(train)

# Prepare features
drop_sensors = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19']
drop_labels = index_names + setting_names + drop_sensors

X_train = train.drop(drop_labels, axis=1)
y_train = X_train.pop('RUL')
y_train_clipped = y_train.clip(upper=125)

print(f"X_train shape: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")

# Fit and save scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'model/scaler.pkl')
print("✓ Scaler saved to model/scaler.pkl")

# Fit and save polynomial features
poly = PolynomialFeatures(2)
X_train_transformed = poly.fit_transform(X_train_scaled)
joblib.dump(poly, 'model/poly.pkl')
print("✓ Polynomial features transformer saved to model/poly.pkl")
print(f"  Polynomial features shape: {X_train_transformed.shape}")

# Fit and save feature selector
svr_temp = SVR(kernel='linear')
svr_temp.fit(X_train_transformed, y_train_clipped)
select_features = SelectFromModel(svr_temp, threshold='mean', prefit=True)
joblib.dump(select_features, 'model/feature_selector.pkl')
print("✓ Feature selector saved to model/feature_selector.pkl")
print(f"  Selected {select_features.get_support().sum()} out of {X_train_transformed.shape[1]} features")

print("\n✅ All preprocessing components saved successfully!")
print("\nYou can now use these in app.py by loading:")
print("  scaler = joblib.load('model/scaler.pkl')")
print("  poly = joblib.load('model/poly.pkl')")
print("  selector = joblib.load('model/feature_selector.pkl')")
