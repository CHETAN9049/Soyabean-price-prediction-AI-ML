import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# 1. App title
# ======================
st.title("Soybean Price Forecaster (2025)")
st.write("Predict soybean wholesale prices for Indian states based on historical data and scenarios.")

# ======================
# 2. Load data
# ======================
@st.cache_data
def load_data():
    df = pd.read_excel('engineered_selected_scaled_fixed.xlsx')
    df['Month'] = pd.to_datetime(df['Month'])
    df.sort_values(['State', 'Month'], inplace=True)
    return df

df = load_data()
states = df['State'].unique()

# ======================
# 3. Sidebar
# ======================
st.sidebar.header("Prediction Settings")
state = st.sidebar.selectbox("Select State", states)
scenario = st.sidebar.selectbox("Select Scenario", ["Baseline", "Rainfall +50%", "Export +20%", "Rainfall -20%"])
rainfall_mod = st.sidebar.slider("Custom Rainfall Change (%)", -50, 50, 0, step=5) / 100 + 1
export_mod = st.sidebar.slider("Custom Export Change (%)", -50, 50, 0, step=5) / 100 + 1

# ======================
# 4. Scenario modifications
# ======================
scenario_mods = {
    "Baseline": {},
    "Rainfall +50%": {"Rainfall Actual (mm)_scaled": 1.5, "Rainfall Actual (mm)_Lag1_scaled": 1.5},
    "Export +20%": {"Export Soybean Meal (Tonnes)_scaled": 1.2},
    "Rainfall -20%": {"Rainfall Actual (mm)_scaled": 0.8, "Rainfall Actual (mm)_Lag1_scaled": 0.8}
}

state_weights = {
    'Andhra Pradesh': {'arimax': 0.5, 'xgb': 0.4, 'mlp': 0.05, 'huber': 0.05},
    'Chhattisgarh': {'arimax': 0.6, 'xgb': 0.2, 'mlp': 0.1, 'huber': 0.1},
    'Gujarat': {'arimax': 0.6, 'xgb': 0.35, 'mlp': 0.025, 'huber': 0.025},
    'Karnataka': {'arimax': 0.7, 'xgb': 0.05, 'mlp': 0.05, 'huber': 0.2},
    'Madhya Pradesh': {'arimax': 0.65, 'xgb': 0.15, 'mlp': 0.1, 'huber': 0.1},
    'Maharashtra': {'arimax': 0.55, 'xgb': 0.4, 'mlp': 0.025, 'huber': 0.025},
    'Manipur': {'arimax': 0.1, 'xgb': 0.55, 'mlp': 0.1, 'huber': 0.25},
    'Nagaland': {'arimax': 0.0, 'xgb': 0.65, 'mlp': 0.15, 'huber': 0.2},
    'Rajasthan': {'arimax': 0.6, 'xgb': 0.36, 'mlp': 0.02, 'huber': 0.02},
    'Tamil Nadu': {'arimax': 0.1, 'xgb': 0.6, 'mlp': 0.1, 'huber': 0.2},
    'Telangana': {'arimax': 0.65, 'xgb': 0.3, 'mlp': 0.03, 'huber': 0.02},
    'Uttar Pradesh': {'arimax': 0.75, 'xgb': 0.15, 'mlp': 0.05, 'huber': 0.05},
    'Uttarakhand': {'arimax': 0.6, 'xgb': 0.25, 'mlp': 0.1, 'huber': 0.05},
}

state_arima_orders = {
    'Andhra Pradesh': (2,1,2), 'Chhattisgarh': (2,1,2), 'Gujarat': (1,0,0),
    'Karnataka': (2,1,2), 'Madhya Pradesh': (2,1,1), 'Maharashtra': (2,1,1),
    'Manipur': (1,1,0), 'Nagaland': (0,1,0), 'Rajasthan': (2,1,2),
    'Tamil Nadu': (1,1,0), 'Telangana': (2,1,2), 'Uttar Pradesh': (1,1,1),
    'Uttarakhand': (2,1,1),
}

state_xgb_params = {
    'Andhra Pradesh': {'n_estimators': 180, 'learning_rate': 0.05, 'max_depth': 3, 'reg_lambda': 2.0},
    'Chhattisgarh': {'n_estimators': 150, 'learning_rate': 0.06, 'max_depth': 3, 'reg_lambda': 2.2},
    'Gujarat': {'n_estimators': 130, 'learning_rate': 0.05, 'max_depth': 3, 'reg_lambda': 2.5},
    'Karnataka': {'n_estimators': 220, 'learning_rate': 0.03, 'max_depth': 5, 'reg_lambda': 1.5},
    'Madhya Pradesh': {'n_estimators': 200, 'learning_rate': 0.04, 'max_depth': 4, 'reg_lambda': 2.3},
    'Maharashtra': {'n_estimators': 200, 'learning_rate': 0.04, 'max_depth': 4, 'reg_lambda': 2.2},
    'Manipur': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4, 'reg_lambda': 2.0},
    'Nagaland': {'n_estimators': 130, 'learning_rate': 0.06, 'max_depth': 3, 'reg_lambda': 2.1},
    'Rajasthan': {'n_estimators': 210, 'learning_rate': 0.05, 'max_depth': 4, 'reg_lambda': 2.0},
    'Tamil Nadu': {'n_estimators': 160, 'learning_rate': 0.05, 'max_depth': 3, 'reg_lambda': 2.2},
    'Telangana': {'n_estimators': 230, 'learning_rate': 0.03, 'max_depth': 5, 'reg_lambda': 1.7},
    'Uttar Pradesh': {'n_estimators': 170, 'learning_rate': 0.05, 'max_depth': 3, 'reg_lambda': 2.3},
    'Uttarakhand': {'n_estimators': 180, 'learning_rate': 0.04, 'max_depth': 4, 'reg_lambda': 1.9},
}

state_mlp_params = {
    'Andhra Pradesh': {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0002, 'alpha': 0.15, 'max_iter': 2000},
    'Chhattisgarh': {'hidden_layer_sizes': (8,8), 'learning_rate_init': 0.001, 'alpha': 0.05, 'max_iter': 2000},
    'Gujarat': {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0003, 'alpha': 0.15, 'max_iter': 2000},
    'Karnataka': {'hidden_layer_sizes': (6,6), 'learning_rate_init': 0.0005, 'alpha': 0.1, 'max_iter': 2000},
    'Madhya Pradesh': {'hidden_layer_sizes': (8,8), 'learning_rate_init': 0.0004, 'alpha': 0.05, 'max_iter': 2000},
    'Maharashtra': {'hidden_layer_sizes': (4,4), 'learning_rate_init': 0.0002, 'alpha': 0.2, 'max_iter': 2000},
    'Manipur': {'hidden_layer_sizes': (3,3), 'learning_rate_init': 0.00015, 'alpha': 0.2, 'max_iter': 2000},
    'Nagaland': {'hidden_layer_sizes': (4,4), 'learning_rate_init': 0.0003, 'alpha': 0.15, 'max_iter': 2000},
    'Rajasthan': {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0002, 'alpha': 0.15, 'max_iter': 2000},
    'Tamil Nadu': {'hidden_layer_sizes': (4,4), 'learning_rate_init': 0.0003, 'alpha': 0.15, 'max_iter': 2000},
    'Telangana': {'hidden_layer_sizes': (4,), 'learning_rate_init': 0.0002, 'alpha': 0.25, 'max_iter': 2000},
    'Uttar Pradesh': {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0003, 'alpha': 0.15, 'max_iter': 2000},
    'Uttarakhand': {'hidden_layer_sizes': (4,4), 'learning_rate_init': 0.0002, 'alpha': 0.15, 'max_iter': 2000},
}

state_huber_params = {
    'Andhra Pradesh': {'epsilon': 1.3, 'alpha': 0.0005},
    'Chhattisgarh': {'epsilon': 1.35, 'alpha': 0.0008},
    'Gujarat': {'epsilon': 1.25, 'alpha': 0.001},
    'Karnataka': {'epsilon': 1.2, 'alpha': 0.0007},
    'Madhya Pradesh': {'epsilon': 1.3, 'alpha': 0.0006},
    'Maharashtra': {'epsilon': 1.3, 'alpha': 0.0008},
    'Manipur': {'epsilon': 1.5, 'alpha': 0.0005},
    'Nagaland': {'epsilon': 1.45, 'alpha': 0.0006},
    'Rajasthan': {'epsilon': 1.35, 'alpha': 0.0007},
    'Tamil Nadu': {'epsilon': 1.4, 'alpha': 0.0007},
    'Telangana': {'epsilon': 1.25, 'alpha': 0.0008},
    'Uttar Pradesh': {'epsilon': 1.35, 'alpha': 0.0006},
    'Uttarakhand': {'epsilon': 1.4, 'alpha': 0.0007},
}

# ======================
# 6. Prepare data for state
# ======================
df_state = df[df['State'] == state].copy()
df_state.set_index('Month', inplace=True)
df_state = df_state.asfreq('MS').sort_index()

X_cols = [col for col in df_state.columns if col not in ['Soybean Prices (₹/qtl)', 'State']]
X = df_state[X_cols]
y = df_state['Soybean Prices (₹/qtl)']

train_mask = df_state.index.year < 2024
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

future_dates = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
X_future = X[-12:].copy()
X_future.index = future_dates

# Apply scenario modifications
mods = scenario_mods.get(scenario, {})
if rainfall_mod != 1:
    mods.update({"Rainfall Actual (mm)_scaled": rainfall_mod, "Rainfall Actual (mm)_Lag1_scaled": rainfall_mod})
if export_mod != 1:
    mods.update({"Export Soybean Meal (Tonnes)_scaled": export_mod})

for feature, multiplier in mods.items():
    if feature in X_future.columns:
        X_future[feature] *= multiplier

# ======================
# 7. Train models (state-specific)
# ======================
exog_cols = [col for col in ['Rainfall Actual (mm)_Lag1_scaled', 'Rainfall Actual (mm)_scaled'] if col in X_train.columns]
exog_train = X_train[exog_cols] if exog_cols else None
exog_test = X_test[exog_cols] if exog_cols else None
exog_future = X_future[exog_cols] if exog_cols else None

# ARIMA
try:
    arima_order = state_arima_orders[state]  # state-specific
    arimax_model = ARIMA(y_train, exog=exog_train, order=arima_order).fit()
except Exception as e:
    st.error(f"ARIMAX failed for {state}: {e}")
    arimax_model = None

# XGBoost
xgb_params = state_xgb_params[state]  # state-specific
xgb_model = XGBRegressor(**xgb_params, random_state=42)
xgb_model.fit(X_train, y_train)

# MLP
mlp_params = state_mlp_params[state]  # state-specific
mlp_model = MLPRegressor(**mlp_params, random_state=42)
mlp_model.fit(X_train, y_train)

# Huber
huber_params = state_huber_params[state]  # state-specific
huber_model = HuberRegressor(**huber_params).fit(X_train, y_train)

# ======================
# 8. Predictions (state-specific ensemble)
# ======================
arimax_pred = arimax_model.forecast(steps=len(X_future), exog=exog_future) if arimax_model else np.full(len(X_future), np.nan)
xgb_pred = xgb_model.predict(X_future)
mlp_pred = mlp_model.predict(X_future)
huber_pred = huber_model.predict(X_future)

# Use state-specific ensemble weights
weights = state_weights[state]
ensemble_pred = (weights['arimax'] * arimax_pred +
                 weights['xgb'] * xgb_pred +
                 weights['mlp'] * mlp_pred +
                 weights['huber'] * huber_pred)

# ======================
# 9. Validate on 2024
# ======================
arimax_test = arimax_model.forecast(steps=len(y_test), exog=exog_test) if arimax_model else np.full(len(y_test), np.nan)
xgb_test = xgb_model.predict(X_test)
mlp_test = mlp_model.predict(X_test)
huber_test = huber_model.predict(X_test)
ensemble_test = (weights['arimax']*arimax_test + 
                 weights['xgb']*xgb_test + 
                 weights['mlp']*mlp_test + 
                 weights['huber']*huber_test)

def evaluate(y_true, y_pred):
    if np.all(np.isnan(y_pred)):
        return [np.nan, np.nan, np.nan, np.nan]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return [rmse, mae, mape, r2]

if len(y_test) > 0:
    rmse, mae, mape, r2 = evaluate(y_test, ensemble_test)
    st.subheader("Validation Metrics (2024)")
    st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}")

# ======================
# 10. Display predictions
# ======================
st.subheader(f"2025 Price Forecast for {state} ({scenario})")
pred_df = pd.DataFrame({"Month": X_future.index, "Predicted Price (₹/qtl)": ensemble_pred})
st.dataframe(pred_df)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X_future.index, ensemble_pred, label="Predicted", marker='o', linewidth=2, color='brown')
ax.set_title(f"2025 Soybean Price Forecast - {state} ({scenario})")
ax.set_xlabel("Month")
ax.set_ylabel("Price (₹/qtl)")
ax.tick_params(axis='x', rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
st.pyplot(fig)

# Historical comparison
if len(y_test) > 0:
    st.subheader("Historical Comparison (2024)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label="Actual", marker='o', color='black')
    ax.plot(y_test.index, ensemble_test, label="Predicted", marker='o', linestyle='--', color='brown')
    ax.set_title(f"2024 Actual vs Predicted - {state}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (₹/qtl)")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig)
