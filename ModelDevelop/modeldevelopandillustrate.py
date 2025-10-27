import os
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure output directory exists
os.makedirs("model_outputs", exist_ok=True)

# Load dataset
try:
    df = pd.read_excel('engineered_selected_scaled_fixed.xlsx')
except FileNotFoundError:
    raise FileNotFoundError("Dataset 'engineered_selected_scaled_fixed.xlsx' not found in project directory.")

df['Month'] = pd.to_datetime(df['Month'])
df.sort_values(['State', 'Month'], inplace=True)

if 'State' not in df.columns:
    raise ValueError("'State' column missing in dataset.")

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

# ============================
# ✅ Modeling Loop (No Change in Logic)
# ============================
results = []
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

for state in df['State'].unique():
    print(f"\nProcessing state: {state}")
    df_state = df[df['State'] == state].copy()
    df_state.set_index('Month', inplace=True)
    df_state = df_state.asfreq('MS').sort_index()

    X_cols = [col for col in df_state.columns if col not in ['Soybean Prices (₹/qtl)', 'State']]
    X = df_state[X_cols]
    y = df_state['Soybean Prices (₹/qtl)']

    train_mask = df_state.index.year < 2024
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    if len(X_train) < 10 or len(X_test) < 1:
        print(f"Skipping {state}: Insufficient data.")
        continue

    # Scaling for MLP
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    # ----- ARIMAX -----
    exog_cols = [c for c in ['Rainfall Actual (mm)_Lag1_scaled', 'Rainfall Actual (mm)_scaled'] if c in X_train.columns]
    try:
        order = state_arima_orders.get(state, (1,1,1))
        arimax_model = ARIMA(y_train, exog=X_train[exog_cols], order=order)
        arimax_result = arimax_model.fit()
        arimax_preds = arimax_result.forecast(steps=len(y_test), exog=X_test[exog_cols])
    except:
        arimax_preds = np.full(len(y_test), np.nan)

    # ----- XGB -----
    p = state_xgb_params[state]
    xgb_model = XGBRegressor(**p, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # ----- MLP -----
    p = state_mlp_params[state]
    mlp = MLPRegressor(
        hidden_layer_sizes=p['hidden_layer_sizes'],
        learning_rate_init=p['learning_rate_init'],
        alpha=p['alpha'],
        learning_rate='adaptive',
        solver='adam',
        max_iter=3000,
        early_stopping=True,
        random_state=42,
        tol=1e-5
    )
    mlp.fit(X_train_scaled, y_train_scaled)
    mlp_preds_scaled = mlp.predict(X_test_scaled)
    mlp_preds = target_scaler.inverse_transform(mlp_preds_scaled.reshape(-1, 1)).flatten()

    # ----- Huber -----
    p = state_huber_params[state]
    huber = HuberRegressor(**p)
    huber.fit(X_train, y_train)
    huber_preds = huber.predict(X_test)

    # ----- Ensemble -----
    w = state_weights[state]
    ensemble_preds = (
        w['arimax'] * arimax_preds +
        w['xgb'] * xgb_preds +
        w['mlp'] * mlp_preds +
        w['huber'] * huber_preds
    )

    # ----- Metrics -----
    def eval_model(name, y_true, y_pred):
        return [
            state, name,
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            r2_score(y_true, y_pred)
        ]

    results.extend([
        eval_model('ARIMAX', y_test, arimax_preds),
        eval_model('XGB', y_test, xgb_preds),
        eval_model('MLP', y_test, mlp_preds),
        eval_model('Huber', y_test, huber_preds),
        eval_model('Ensemble', y_test, ensemble_preds)
    ])

    # ----- Plot -----
    plt.figure(figsize=(14, 8))
    plt.plot(y_test.index, y_test, label='Actual', color='black', marker='o')
    plt.plot(y_test.index, arimax_preds, label='ARIMAX', color='blue', linestyle='--')
    plt.plot(y_test.index, xgb_preds, label='XGB', color='green', linestyle='-.')
    plt.plot(y_test.index, mlp_preds, label='MLP', color='red', linestyle=':')
    plt.plot(y_test.index, huber_preds, label='Huber', color='purple', linestyle='-')
    plt.plot(y_test.index, ensemble_preds, label='Ensemble', color='brown', linewidth=2)
    plt.title(f"Soybean Price Forecast (2024) - {state}")
    plt.xlabel("Month")
    plt.ylabel("Price (₹/qtl)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"model_outputs/forecast_comparison_{state.replace(' ', '_')}.png", dpi=300)
    plt.close()

# ----- Save -----
res = pd.DataFrame(results, columns=["State", "Model", "RMSE", "MAE", "MAPE (%)", "R²"])
res.to_excel("model_outputs/model_comparison_by_state.xlsx", index=False)
print("\n✅ All state forecasts and metrics saved successfully.")