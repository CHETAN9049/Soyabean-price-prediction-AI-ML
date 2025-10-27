import os
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 1. Load Data
# ==========================================================
df = pd.read_excel('engineered_selected_scaled_fixed.xlsx')
df['Month'] = pd.to_datetime(df['Month'])
df.sort_values(['State', 'Month'], inplace=True)

if 'State' not in df.columns:
    raise ValueError("'State' column missing in dataset.")

# ==========================================================
# 2. State-specific configurations
# ==========================================================

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

# ==========================================================
# 3. Backtesting Loop
# ==========================================================

window_size = 48
step_size = 1
results = []

output_dir = "backtest_png_results"
os.makedirs(output_dir, exist_ok=True)

for state in df['State'].unique():
    print(f"\nBacktesting for state: {state}")
    df_state = df[df['State'] == state].copy()
    df_state.set_index('Month', inplace=True)
    df_state = df_state.asfreq('MS').sort_index()

    X_cols = [c for c in df_state.columns if c not in ['Soybean Prices (₹/qtl)', 'State']]
    X = df_state[X_cols]
    y = df_state['Soybean Prices (₹/qtl)']

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    backtest_preds, backtest_true, backtest_months = [], [], []

    for i in range(window_size, len(df_state) - step_size + 1):
        train_idx = df_state.index[:i]
        test_idx = df_state.index[i:i + step_size]

        X_train, X_test = X_scaled.loc[train_idx], X_scaled.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # Model setup
        arima_order = state_arima_orders[state]
        xgb_params = state_xgb_params[state]
        mlp_params = state_mlp_params[state]
        huber_params = state_huber_params[state]
        weights = state_weights[state]

        try:
            arimax_model = ARIMA(y_train, order=arima_order).fit()
            arimax_pred = arimax_model.forecast(steps=step_size)
        except:
            arimax_pred = np.full(step_size, np.nan)

        try:
            xgb_model = XGBRegressor(**xgb_params, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
        except:
            xgb_pred = np.full(step_size, np.nan)

        try:
            mlp_model = MLPRegressor(**mlp_params, random_state=42)
            mlp_model.fit(X_train, y_train)
            mlp_pred = mlp_model.predict(X_test)
        except:
            mlp_pred = np.full(step_size, np.nan)

        try:
            huber_model = HuberRegressor(**huber_params)
            huber_model.fit(X_train, y_train)
            huber_pred = huber_model.predict(X_test)
        except:
            huber_pred = np.full(step_size, np.nan)

        # Weighted ensemble
        ensemble_pred = (
            weights['arimax'] * arimax_pred +
            weights['xgb'] * xgb_pred +
            weights['mlp'] * mlp_pred +
            weights['huber'] * huber_pred
        )

        backtest_preds.extend(ensemble_pred)
        backtest_true.extend(y_test)
        backtest_months.extend(test_idx)

    # Evaluation
    backtest_preds = np.array(backtest_preds)
    backtest_true = np.array(backtest_true)

    if len(backtest_preds) > 0 and not np.all(np.isnan(backtest_preds)):
        rmse = np.sqrt(mean_squared_error(backtest_true, backtest_preds))
        mae = mean_absolute_error(backtest_true, backtest_preds)
        mape = np.mean(np.abs((backtest_true - backtest_preds) / backtest_true)) * 100
        r2 = r2_score(backtest_true, backtest_preds)
        results.append([state, rmse, mae, mape, r2])

        # ======================================================
        # 4. Save diagram as PNG for each state
        # ======================================================
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_months, backtest_true, label="Actual", color='black', linewidth=2)
        plt.plot(backtest_months, backtest_preds, label="Ensemble Forecast", linestyle='--', color='brown', linewidth=2)
        plt.title(f"Soybean Price Backtest Forecast - {state}", fontsize=15)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Soybean Price (₹/qtl)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{state.replace(' ', '_')}_backtest.png", dpi=300)
        plt.close()

# ==========================================================
# 5. Save metrics to Excel
# ==========================================================
results_df = pd.DataFrame(results, columns=['State', 'RMSE', 'MAE', 'MAPE (%)', 'R²'])
results_df.to_excel('backtest_results_by_state.xlsx', index=False)

print("\n✅ Backtesting completed successfully.")
print("➡ Results saved to: backtest_results_by_state.xlsx")
print(f"➡ PNG charts saved in: {output_dir}/")
