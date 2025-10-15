import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------
# Config
# ----------------------------
INPUT_FILE = "Final_Merged_Soybean_Dataset_20250826_110201.xlsx"
OUT_DIR = "."
TARGET = "Soybean Prices (₹/qtl)"
STATE_COL = "State"
MONTH_COL = "Month"

LAG_COLS = [
    TARGET,
    "Yield (In Kg./Hectare)",
    "Production (In '000 Tonne)",
    "Market Arrivals (Tonnes)",
    "Rainfall Actual (mm)",
]
LAGS = [1, 3, 6, 12]
RF_SELECTOR_THRESHOLD = 0.0025  # Lowered to retain state dummies
MIN_KEEP = 15  # Increased to ensure state features
VIF_THRESHOLD = 5  # Stricter for stability
PROTECTED_STATES = ["State_Manipur", "State_Rajasthan"]  # EDA-driven high-price states

# ----------------------------
# Load
# ----------------------------
print("Loading:", INPUT_FILE)
df = pd.read_excel(INPUT_FILE)
if MONTH_COL not in df.columns:
    raise ValueError(f"'{MONTH_COL}' column not found in input file.")
if STATE_COL not in df.columns:
    raise ValueError(f"'{STATE_COL}' column not found in input file.")

df[MONTH_COL] = pd.to_datetime(df[MONTH_COL])
has_state = True  # Confirmed by input

# Sort
sort_cols = [STATE_COL, MONTH_COL]
df = df.sort_values(sort_cols).reset_index(drop=True)

# ----------------------------
# Calendar & season flags
# ----------------------------
df["Year"] = df[MONTH_COL].dt.year
df["Month_Num"] = df[MONTH_COL].dt.month
df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
df["Mid_Year_Peak"] = df["Month_Num"].isin([7, 8, 9]).astype(int)  # Aligned with EDA
df["Harvest_Season"] = df["Month_Num"].isin([10, 11]).astype(int)

# ----------------------------
# Add state-aware lags
# ----------------------------
print("Creating state-aware lags...")
frames = []
for st, g in df.groupby(STATE_COL):
    g2 = g.copy().sort_values(MONTH_COL)
    for col in LAG_COLS:
        if col in g2.columns:
            for lag in LAGS:
                g2[f"{col}_Lag{lag}"] = g2[col].shift(lag)
    if TARGET in g2.columns:
        g2["% Change (Over Previous Month)"] = g2[TARGET].pct_change() * 100
        g2["% Change (Over Previous Year)"] = g2[TARGET].pct_change(12) * 100
    frames.append(g2)
df = pd.concat(frames, axis=0).sort_values(sort_cols).reset_index(drop=True)

# ----------------------------
# Derived ratios & interactions
# ----------------------------
print("Creating derived features & interactions...")
if "Import Soyabean Oil (Tonnes)" in df.columns and "Export Soybean Meal (Tonnes)" in df.columns:
    df["Import_Export_Ratio"] = df["Import Soyabean Oil (Tonnes)"] / (df["Export Soybean Meal (Tonnes)"] + 1e-6)
if "Rainfall Departure (%)" in df.columns and "Yield (In Kg./Hectare)" in df.columns:
    df["Rain_Yield_Interact"] = df["Rainfall Departure (%)"] * df["Yield (In Kg./Hectare)"]
if "Market Arrivals (Tonnes)" in df.columns and "Export Soybean Meal (Tonnes)" in df.columns:
    df["Arrivals_Export_Interact"] = df["Market Arrivals (Tonnes)"] * df["Export Soybean Meal (Tonnes)"]

# ----------------------------
# Safe state-wise imputation
# ----------------------------
print("Imputing missing values state-wise (ffill/bfill then mean)...")
def impute_state_group(g):
    g = g.sort_values(MONTH_COL)
    g = g.ffill().bfill()
    num_cols = g.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if g[c].isna().any():
            g[c] = g[c].fillna(g[c].mean())
    return g

frames = []
for st, g in df.groupby(STATE_COL):
    frames.append(impute_state_group(g.copy()))
df = pd.concat(frames, axis=0).sort_values(sort_cols).reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

# Save full engineered dataset
full_path = os.path.join(OUT_DIR, "engineered_full_fixed.xlsx")
df.to_excel(full_path, index=False)
print("Saved full engineered (pre-dummy/impute) ->", full_path, " shape=", df.shape)

# ----------------------------
# One-hot encode State, but keep original State column
# ----------------------------
dummies = pd.get_dummies(df[STATE_COL], prefix="State", drop_first=True)
df = pd.concat([df, dummies], axis=1)
# Do NOT drop the original State column

# ----------------------------
# Prepare X for VIF pruning
# ----------------------------
drop_cols_base = [TARGET, MONTH_COL]
X_cols = [c for c in df.columns if c not in drop_cols_base]
X_num = df[X_cols].select_dtypes(include=[np.number]).copy()

# Compute VIF
def compute_vif(Xframe):
    vif_list = []
    Xfix = Xframe.copy()
    for c in Xfix.columns:
        if Xfix[c].std() == 0:
            Xfix[c] = Xfix[c] + np.random.normal(0, 1e-9, size=len(Xfix))
    for i, col in enumerate(Xfix.columns):
        try:
            v = variance_inflation_factor(Xfix.values, i)
        except Exception:
            v = np.inf
        vif_list.append((col, v))
    return pd.DataFrame(vif_list, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False)

# Iterative VIF drop (>5)
print("Running iterative VIF pruning...")
vif_drops = []
max_iter = 15
for _ in range(max_iter):
    vif_df = compute_vif(X_num)
    high = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
    if high.empty:
        break
    to_drop = high.iloc[0]["Feature"]
    if to_drop in PROTECTED_STATES:
        continue  # Skip protected states
    vif_val = float(high.iloc[0]["VIF"])
    vif_drops.append((to_drop, vif_val))
    X_num = X_num.drop(columns=[to_drop])
print("VIF drops (feature, VIF):")
for f, v in vif_drops:
    print(" -", f, f"{v:.2f}")

# Build VIF-pruned df (keep State column)
df_vif = pd.concat([df[drop_cols_base + [STATE_COL]], X_num], axis=1)
vif_path = os.path.join(OUT_DIR, "engineered_vif_pruned_fixed.xlsx")
df_vif.to_excel(vif_path, index=False)
print("Saved VIF-pruned dataset ->", vif_path, " shape=", df_vif.shape)

# ----------------------------
# Model-based selection (RandomForest)
# ----------------------------
X_raw = df_vif.drop(columns=drop_cols_base + [STATE_COL])  # Exclude State from features
y_raw = df_vif[TARGET].copy()

# Time split: 2023-2024 as test
max_year = df_vif[MONTH_COL].dt.year.max()
cutoff = pd.Timestamp(max_year - 1, 1, 1)  # 2023-01-01
train_mask = df_vif[MONTH_COL] < cutoff
test_mask = ~train_mask

X_train = X_raw.loc[train_mask]
X_test = X_raw.loc[test_mask]
y_train = np.log1p(y_raw.loc[train_mask])  # Log transform target
y_test = np.log1p(y_raw.loc[test_mask])

print("Fitting RF to compute importances...")
rf_full = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf_full.fit(X_train.fillna(0), y_train)

importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_full.feature_importances_
}).sort_values("Importance", ascending=False)

imp_path = os.path.join(OUT_DIR, "feature_importances_fixed.csv")
importances.to_csv(imp_path, index=False)
print("Saved feature importances ->", imp_path)

# SelectFromModel with protected states
selector = SelectFromModel(rf_full, threshold=RF_SELECTOR_THRESHOLD, prefit=True)
support = selector.get_support()
selected_cols = X_train.columns[support].tolist()
selected_cols = sorted(list(set(selected_cols) | set(PROTECTED_STATES) & set(X_train.columns)))

if len(selected_cols) < MIN_KEEP:
    top_fallback = importances["Feature"].head(MIN_KEEP).tolist()
    selected_cols = sorted(list(set(selected_cols) | set(top_fallback) & set(X_train.columns)))

print(f"Selected {len(selected_cols)} features (threshold={RF_SELECTOR_THRESHOLD}):")
for c in selected_cols:
    print(" -", c)

# ----------------------------
# Scale selected features
# ----------------------------
scaler = StandardScaler()
scaler.fit(X_train[selected_cols].fillna(0))

X_selected_all = X_raw[selected_cols].copy()
X_selected_scaled_all = pd.DataFrame(
    scaler.transform(X_selected_all.fillna(0)),
    columns=[f"{c}_scaled" for c in selected_cols],
    index=X_selected_all.index
)

# Save selected datasets (keep State column)
df_selected = pd.concat([df_vif[[MONTH_COL, TARGET, STATE_COL]], X_selected_all], axis=1)
sel_path = os.path.join(OUT_DIR, "engineered_selected_fixed.xlsx")
df_selected.to_excel(sel_path, index=False)

df_selected_scaled = pd.concat([df_vif[[MONTH_COL, TARGET, STATE_COL]], X_selected_scaled_all], axis=1)
sel_scaled_path = os.path.join(OUT_DIR, "engineered_selected_scaled_fixed.xlsx")
df_selected_scaled.to_excel(sel_scaled_path, index=False)

# Optional: Save per-state datasets
print("Saving per-state datasets...")
state_dummy_cols = [col for col in df_selected_scaled.columns if col.startswith("State_")]
for state_col in state_dummy_cols:
    state_df = df_selected_scaled[df_selected_scaled[state_col] == 1]
    state_name = state_col.replace("State_", "")
    state_path = os.path.join(OUT_DIR, f"engineered_selected_scaled_{state_name.replace(' ', '_')}.xlsx")
    state_df.to_excel(state_path, index=False)
    print(f" - {state_path}")

print("Saved selected raw ->", sel_path)
print("Saved selected scaled ->", sel_scaled_path)

# ----------------------------
# Sanity-check model
# ----------------------------
X_train_s = X_selected_scaled_all.loc[train_mask]
X_test_s = X_selected_scaled_all.loc[test_mask]

rf_final = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_final.fit(X_train_s.fillna(0), y_train)

if len(X_test_s) > 0 and len(y_test) > 0:
    y_pred = rf_final.predict(X_test_s.fillna(0))
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    try:
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
    except Exception:
        mape = np.nan
    print(f"Holdout performance: RMSE = {rmse:,.2f}, MAPE = {mape*100:.2f}%")
else:
    print("Not enough holdout data for evaluation.")

print("\nDone. Outputs:")
print(" -", full_path)
print(" -", vif_path)
print(" -", imp_path)
print(" -", sel_path)
print(" -", sel_scaled_path)