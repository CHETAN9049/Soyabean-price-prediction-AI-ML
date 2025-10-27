import os
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# 1. Load Data
# ======================
df = pd.read_excel('engineered_selected_scaled_fixed.xlsx')
df['Month'] = pd.to_datetime(df['Month'])
df.sort_values(['State', 'Month'], inplace=True)

if 'State' not in df.columns:
    raise ValueError("'State' column missing in dataset.")

# ======================
# 2. Scenario Analysis Setup
# ======================
scenarios = [
    {"name": "Baseline", "modifications": {}},
    {"name": "Rainfall_+50%", "modifications": {"Rainfall Actual (mm)_scaled": 1.5, "Rainfall Actual (mm)_Lag1_scaled": 1.5}},
    {"name": "Export_+20%", "modifications": {"Export Soybean Meal (Tonnes)_scaled": 1.2}},
    {"name": "Rainfall_-20%", "modifications": {"Rainfall Actual (mm)_scaled": 0.8, "Rainfall Actual (mm)_Lag1_scaled": 0.8}}
]

results = []

# ============================
# ✅ State-Specific Ensemble Weights (Post-Backtesting)
# ============================
state_weights = {
    'Andhra Pradesh': {'arimax': 0.55, 'xgb': 0.35, 'mlp': 0.05, 'huber': 0.05},
    'Chhattisgarh': {'arimax': 0.25, 'xgb': 0.15, 'mlp': 0.35, 'huber': 0.25},
    'Gujarat': {'arimax': 0.6, 'xgb': 0.3, 'mlp': 0.05, 'huber': 0.05},
    'Karnataka': {'arimax': 0.5, 'xgb': 0.35, 'mlp': 0.1, 'huber': 0.05},
    'Madhya Pradesh': {'arimax': 0.5, 'xgb': 0.3, 'mlp': 0.1, 'huber': 0.1},
    'Maharashtra': {'arimax': 0.5, 'xgb': 0.35, 'mlp': 0.1, 'huber': 0.05},
    'Manipur': {'arimax': 0.25, 'xgb': 0.45, 'mlp': 0.15, 'huber': 0.15},
    'Nagaland': {'arimax': 0.25, 'xgb': 0.5, 'mlp': 0.1, 'huber': 0.15},
    'Rajasthan': {'arimax': 0.55, 'xgb': 0.35, 'mlp': 0.05, 'huber': 0.05},
    'Tamil Nadu': {'arimax': 0.3, 'xgb': 0.45, 'mlp': 0.1, 'huber': 0.15},
    'Telangana': {'arimax': 0.55, 'xgb': 0.35, 'mlp': 0.05, 'huber': 0.05},
    'Uttar Pradesh': {'arimax': 0.55, 'xgb': 0.3, 'mlp': 0.1, 'huber': 0.05},
    'Uttarakhand': {'arimax': 0.35, 'xgb': 0.45, 'mlp': 0.1, 'huber': 0.1},
}

# ============================
# ✅ ARIMA Orders
# ============================
state_arima_orders = {
    'Andhra Pradesh': (2,1,2),
    'Chhattisgarh': (2,1,1),
    'Gujarat': (1,1,1),
    'Karnataka': (2,1,2),
    'Madhya Pradesh': (2,1,2),
    'Maharashtra': (2,1,1),
    'Manipur': (0,1,0),
    'Nagaland': (0,1,0),
    'Rajasthan': (2,1,2),
    'Tamil Nadu': (1,1,1),
    'Telangana': (2,1,2),
    'Uttar Pradesh': (1,1,1),
    'Uttarakhand': (1,1,1),
}

# ============================
# ✅ Updated XGBoost Params
# ============================
state_xgb_params = {
    'Andhra Pradesh': {'n_estimators': 200, 'learning_rate': 0.04, 'max_depth': 4, 'reg_lambda': 1.8},
    'Chhattisgarh': {'n_estimators': 180, 'learning_rate': 0.05, 'max_depth': 3, 'reg_lambda': 2.0},
    'Gujarat': {'n_estimators': 180, 'learning_rate': 0.03, 'max_depth': 3, 'reg_lambda': 2.5},
    'Karnataka': {'n_estimators': 250, 'learning_rate': 0.025, 'max_depth': 4, 'reg_lambda': 2.4},
    'Madhya Pradesh': {'n_estimators': 250, 'learning_rate': 0.03, 'max_depth': 4, 'reg_lambda': 2.5},
    'Maharashtra': {'n_estimators': 240, 'learning_rate': 0.03, 'max_depth': 4, 'reg_lambda': 2.4},
    'Manipur': {'n_estimators': 110, 'learning_rate': 0.07, 'max_depth': 3, 'reg_lambda': 2.0},
    'Nagaland': {'n_estimators': 130, 'learning_rate': 0.06, 'max_depth': 3, 'reg_lambda': 2.1},
    'Rajasthan': {'n_estimators': 210, 'learning_rate': 0.05, 'max_depth': 4, 'reg_lambda': 2.0},
    'Tamil Nadu': {'n_estimators': 160, 'learning_rate': 0.05, 'max_depth': 3, 'reg_lambda': 2.2},
    'Telangana': {'n_estimators': 220, 'learning_rate': 0.03, 'max_depth': 4, 'reg_lambda': 2.3},
    'Uttar Pradesh': {'n_estimators': 200, 'learning_rate': 0.035, 'max_depth': 3, 'reg_lambda': 2.3},
    'Uttarakhand': {'n_estimators': 130, 'learning_rate': 0.06, 'max_depth': 3, 'reg_lambda': 2.0},
}

# ============================
# ✅ MLP Params (Fine-Tuned)
# ============================
state_mlp_params = {
    'Andhra Pradesh': {'hidden_layer_sizes': (6,6), 'learning_rate_init': 0.00025, 'alpha': 0.12},
    'Chhattisgarh': {'hidden_layer_sizes': (8,8), 'learning_rate_init': 0.0008, 'alpha': 0.07},
    'Gujarat': {'hidden_layer_sizes': (6,6), 'learning_rate_init': 0.0002, 'alpha': 0.15},
    'Karnataka': {'hidden_layer_sizes': (7,7), 'learning_rate_init': 0.0004, 'alpha': 0.1},
    'Madhya Pradesh': {'hidden_layer_sizes': (8,8), 'learning_rate_init': 0.0003, 'alpha': 0.08},
    'Maharashtra': {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0002, 'alpha': 0.15},
    'Manipur': {'hidden_layer_sizes': (3,3), 'learning_rate_init': 0.0001, 'alpha': 0.25},
    'Nagaland': {'hidden_layer_sizes': (4,4), 'learning_rate_init': 0.0002, 'alpha': 0.2},
    'Rajasthan': {'hidden_layer_sizes': (6,6), 'learning_rate_init': 0.00025, 'alpha': 0.12},
    'Tamil Nadu': {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0002, 'alpha': 0.15},
    'Telangana': {'hidden_layer_sizes': (4,), 'learning_rate_init': 0.0002, 'alpha': 0.25},
    'Uttar Pradesh': {'hidden_layer_sizes': (6,6), 'learning_rate_init': 0.00025, 'alpha': 0.12},
    'Uttarakhand': {'hidden_layer_sizes': (4,4), 'learning_rate_init': 0.0002, 'alpha': 0.15},
}

# ============================
# ✅ Huber Params (Backtested)
# ============================
state_huber_params = {
    'Andhra Pradesh': {'epsilon': 2.3, 'max_iter': 500, 'alpha': 0.00005},
    'Chhattisgarh': {'epsilon': 2.0, 'max_iter': 500, 'alpha': 0.00005},
    'Gujarat': {'epsilon': 1.3, 'max_iter': 500, 'alpha': 0.004},
    'Karnataka': {'epsilon': 1.4, 'max_iter': 500, 'alpha': 0.003},
    'Madhya Pradesh': {'epsilon': 1.5, 'max_iter': 500, 'alpha': 0.003},
    'Maharashtra': {'epsilon': 1.3, 'max_iter': 500, 'alpha': 0.004},
    'Manipur': {'epsilon': 1.1, 'max_iter': 500, 'alpha': 0.009},
    'Nagaland': {'epsilon': 1.1, 'max_iter': 500, 'alpha': 0.009},
    'Rajasthan': {'epsilon': 1.3, 'max_iter': 500, 'alpha': 0.004},
    'Tamil Nadu': {'epsilon': 1.2, 'max_iter': 500, 'alpha': 0.005},
    'Telangana': {'epsilon': 1.2, 'max_iter': 500, 'alpha': 0.009},
    'Uttar Pradesh': {'epsilon': 1.5, 'max_iter': 500, 'alpha': 0.003},
    'Uttarakhand': {'epsilon': 1.4, 'max_iter': 500, 'alpha': 0.003},
}

for state in df['State'].unique():
    print(f"\nScenario analysis for state: {state}")
    df_state = df[df['State'] == state].copy()
    df_state.set_index('Month', inplace=True)
    df_state = df_state.asfreq('MS').sort_index()

    X_cols = [col for col in df_state.columns if col not in ['Soybean Prices (₹/qtl)', 'State']]
    X = df_state[X_cols]
    y = df_state['Soybean Prices (₹/qtl)']

    test_mask = df_state.index.year >= 2024
    X_test = X[test_mask]
    y_test = y[test_mask]

    train_mask = df_state.index.year < 2024
    X_train, y_train = X[train_mask], y[train_mask]

    exog_cols = [col for col in ['Rainfall Actual (mm)_Lag1_scaled', 'Rainfall Actual (mm)_scaled'] if col in X_train.columns]
    exog_train = X_train[exog_cols] if exog_cols else None
    exog_test = X_test[exog_cols] if exog_cols else None

    # ------------------------------
    # ARIMA
    # ------------------------------
    try:
        arima_order = state_arima_orders.get(state, (2,1,2))
        arimax_model = ARIMA(y_train, exog=exog_train, order=arima_order).fit()
    except Exception as e:
        print(f"ARIMAX failed for {state}: {e}")
        arimax_model = None

    # ------------------------------
    # XGBoost
    # ------------------------------
    xgb_params = state_xgb_params.get(state, {'n_estimators':150, 'learning_rate':0.05, 'max_depth':3, 'reg_lambda':2.0})
    xgb_model = XGBRegressor(**xgb_params, random_state=42)
    xgb_model.fit(X_train, y_train)

    # ------------------------------
    # MLP
    # ------------------------------
    mlp_params = state_mlp_params.get(state, {'hidden_layer_sizes': (5,5), 'learning_rate_init': 0.0002, 'alpha': 0.15})
    mlp_model = MLPRegressor(**mlp_params, max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)

    # ------------------------------
    # Huber
    # ------------------------------
    huber_params = state_huber_params.get(state, {'epsilon':2.0, 'max_iter':500, 'alpha':0.00005})
    huber_model = HuberRegressor(**huber_params).fit(X_train, y_train)

    # ------------------------------
    # Scenario Predictions
    # ------------------------------
    weights = state_weights.get(state, {'arimax':0.6, 'xgb':0.3, 'mlp':0.0, 'huber':0.1})

    for scenario in scenarios:
        X_test_scenario = X_test.copy()
        for feature, multiplier in scenario["modifications"].items():
            if feature in X_test_scenario.columns:
                X_test_scenario[feature] *= multiplier

        arimax_pred = arimax_model.forecast(steps=len(y_test), exog=X_test_scenario[exog_cols] if exog_cols and arimax_model else None) if arimax_model else np.full(len(y_test), np.nan)
        xgb_pred = xgb_model.predict(X_test_scenario)
        mlp_pred = mlp_model.predict(X_test_scenario)
        huber_pred = huber_model.predict(X_test_scenario)

        # Align lengths
        min_len = min(map(len, [arimax_pred, xgb_pred, mlp_pred, huber_pred]))
        arimax_pred = np.array(arimax_pred)[-min_len:]
        xgb_pred = np.array(xgb_pred)[-min_len:]
        mlp_pred = np.array(mlp_pred)[-min_len:]
        huber_pred = np.array(huber_pred)[-min_len:]
        y_stack = np.array(y_test)[-min_len:]
        months = y_test.index[-min_len:]

        # Ensemble
        ensemble_pred = (weights['arimax']*arimax_pred + 
                         weights['xgb']*xgb_pred + 
                         weights['mlp']*mlp_pred + 
                         weights['huber']*huber_pred)

        # Evaluation
        def evaluate(name, y_true, y_pred):
            if np.all(np.isnan(y_pred)):
                return [state, name, scenario["name"], np.nan, np.nan, np.nan, np.nan]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)
            return [state, name, scenario["name"], rmse, mae, mape, r2]

        results.append(evaluate("Ensemble", y_stack, ensemble_pred))

        # Plot for non-baseline
        if scenario["name"] != "Baseline":
            plt.figure(figsize=(14, 8))
            plt.plot(months, y_stack, label="Actual", marker='o', linewidth=2, color='black')
            plt.plot(months, ensemble_pred, label=f"Ensemble ({scenario['name']})", linewidth=2, linestyle="--", color='brown')
            plt.title(f"Scenario Analysis: Soybean Price Forecast (2024) - {state} ({scenario['name']})", fontsize=16)
            plt.xlabel("Month")
            plt.ylabel("Price (₹/qtl)")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"scenario_{state.replace(' ', '_')}_{scenario['name'].replace(' ', '_')}.png", dpi=300)
            plt.close()

# ======================
# Save Results
# ======================
results_df = pd.DataFrame(results, columns=["State", "Model", "Scenario", "RMSE", "MAE", "MAPE (%)", "R²"])
print("\nScenario Analysis Results by State:")
print(results_df)
results_df.to_excel("scenario_analysis_results.xlsx", index=False)
print("\nScenario analysis predictions and metrics saved.")
