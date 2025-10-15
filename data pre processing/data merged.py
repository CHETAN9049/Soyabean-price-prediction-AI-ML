import pandas as pd
import os
from datetime import datetime

folder_path = r"C:\Users\cheta\Downloads\clean data\clean data"

if not os.path.exists(folder_path):
    raise FileNotFoundError(f"Directory not found: {folder_path}")

def load_file(file_path, is_excel=False):
    try:
        if is_excel:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")

files = {
    "groundnut": ("Groundnut_Prices_Cleaned_Origin.csv", False),
    "mustard": ("Mustard_Prices_Cleaned_Origin.csv", False),
    "soy_arrivals": ("Soyabean_Arrivals_Cleaned_Origin.csv", False),
    "soy_prices": ("Soyabean_Prices_Cleaned_Origin.csv", False),
    "imports": ("soyabean_monthly_imports_Jan2020_Jul2025.xlsx", True),
    "exports": ("exportoilmeal.xlsx", True),
    "rainfall": ("processed_rainfall.xlsx", True),
    "msp": ("monthly_msp_output.xlsx", True),
    "area_prod": ("cleaned_soybean_data_complete.csv", False)
}

dataframes = {name: load_file(os.path.join(folder_path, filename), is_excel)
              for name, (filename, is_excel) in files.items()}

specified_states = [
    'Andhra Pradesh', 'Chhattisgarh', 'Gujarat', 'Karnataka', 'Madhya Pradesh',
    'Maharashtra', 'Manipur', 'Nagaland', 'Rajasthan', 'Tamil Nadu',
    'Telangana', 'Uttar Pradesh', 'Uttarakhand'
]

state_corrections = {
    "Chattisgarh": "Chhattisgarh",
    "Uttrakhand": "Uttarakhand",
    "MadhyaPradesh": "Madhya Pradesh",
    "Madhya pradesh": "Madhya Pradesh",
    "Uttaranchal": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "MP": "Madhya Pradesh",
    "Orissa": "Odisha"
}

def normalize_states(df):
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.strip()
        df["State"] = df["State"].replace(state_corrections)
    return df

for name in dataframes:
    dataframes[name] = normalize_states(dataframes[name])

area_df = dataframes["area_prod"].copy()
area_df.columns = ['State', 'Year', 'Area (In \'000 Hectare)', 
                   'Production (In \'000 Tonne)', 'Yield (In Kg./Hectare)']
area_df = area_df[area_df['State'].isin(specified_states)]

area_df['Year'] = pd.to_datetime(area_df['Year'], format='%Y-%m').dt.year

years = range(2020, 2025)
full_grid = pd.DataFrame([(state, year) for state in specified_states for year in years],
                         columns=['State', 'Year'])

area_annual = full_grid.merge(area_df, on=['State', 'Year'], how='left').fillna(0)

monthly_data = []
for _, row in area_annual.iterrows():
    for month in range(1, 13):
        month_str = f"{int(row['Year'])}-{month:02d}"
        monthly_data.append({
            'State': row['State'],
            'Month': month_str,
            'Area (In \'000 Hectare)': row['Area (In \'000 Hectare)'],
            'Production (In \'000 Tonne)': row['Production (In \'000 Tonne)'],
            'Yield (In Kg./Hectare)': row['Yield (In Kg./Hectare)']
        })

dataframes["area_prod_monthly"] = pd.DataFrame(monthly_data)

def to_month(df, col):
    if col not in df.columns:
        return df
    df["Month"] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m")
    return df

for name in ["groundnut", "mustard", "soy_arrivals", "soy_prices", "imports", "msp"]:
    dataframes[name] = to_month(dataframes[name], "Date")

dataframes["exports"]["Month"] = pd.to_datetime(
    dataframes["exports"]["Year"].astype(str) + "-" + dataframes["exports"]["Month"].astype(str),
    format="%Y-%B", errors="coerce"
).dt.strftime("%Y-%m")

dataframes["rainfall"]["Month"] = pd.to_datetime(
    dataframes["rainfall"]["Year"].astype(str) + "-" + dataframes["rainfall"]["Month"].astype(str),
    format="%Y-%b", errors="coerce"
).dt.strftime("%Y-%m")

dataframes["groundnut"] = dataframes["groundnut"].rename(columns={"Price": "Groundnut (Peanut) Prices (₹/qtl)"})
dataframes["mustard"] = dataframes["mustard"].rename(columns={"Price": "Rapeseed (Mustard) Prices (₹/qtl)"})
dataframes["soy_prices"] = dataframes["soy_prices"].rename(columns={
    "Price": "Soybean Prices (₹/qtl)",
    "ChangePrevMonth": "% Change (Over Previous Month)",
    "ChangePrevYear": "% Change (Over Previous Year)"
})
dataframes["soy_arrivals"] = dataframes["soy_arrivals"].rename(columns={
    "Market Arrivals (given year)": "Market Arrivals (Tonnes)",
    "Market Arrivals (previous year)": "Prev Market Arrivals (Tonnes)",
    "% (+/-) WRT(previous year)": "WRT (previous year)",
    "State%": "State %"
})
dataframes["imports"] = dataframes["imports"].rename(columns={"Soyabean Oil Monthly": "Import Soyabean Oil (Tonnes)"})

if "Soyabean Meal" in dataframes["exports"].columns:
    dataframes["exports"] = dataframes["exports"].rename(columns={"Soyabean Meal": "Export Soybean Meal (Tonnes)"})
else:
    for col in dataframes["exports"].columns:
        if "soy" in col.lower() and "meal" in col.lower():
            dataframes["exports"] = dataframes["exports"].rename(columns={col: "Export Soybean Meal (Tonnes)"})
            break

dataframes["msp"] = dataframes["msp"].rename(columns={"Soybean MSP Monthly": "MSP (₹/qtl)"})

for name in ["groundnut", "mustard", "soy_prices", "soy_arrivals"]:
    df = dataframes[name]
    if 'State' in df.columns and 'Month' in df.columns:
        if df.duplicated(subset=["State", "Month"]).any():
            dataframes[name] = df.groupby(["State", "Month"]).mean(numeric_only=True).reset_index()

final = dataframes["rainfall"][["State", "Month", "Rainfall Actual (mm)", "Rainfall Normal (mm)", "Rainfall Departure (%)"]]

merge_keys = ["State", "Month"]
for name, columns in [
    ("soy_arrivals", ["State", "Month", "Market Arrivals (Tonnes)", "Prev Market Arrivals (Tonnes)", "WRT (previous year)", "State %"]),
    ("soy_prices", ["State", "Month", "Soybean Prices (₹/qtl)", "% Change (Over Previous Month)", "% Change (Over Previous Year)"]),
    ("groundnut", ["State", "Month", "Groundnut (Peanut) Prices (₹/qtl)"]),
    ("mustard", ["State", "Month", "Rapeseed (Mustard) Prices (₹/qtl)"]),
    ("area_prod_monthly", ["State", "Month", "Area (In '000 Hectare)", "Production (In '000 Tonne)", "Yield (In Kg./Hectare)"])
]:
    if set(columns).issubset(dataframes[name].columns):
        final = final.merge(dataframes[name][columns], on=merge_keys, how="outer")

for name, columns in [
    ("imports", ["Month", "Import Soyabean Oil (Tonnes)"]),
    ("exports", ["Month", "Export Soybean Meal (Tonnes)"]),
    ("msp", ["Month", "MSP (₹/qtl)"])
]:
    final = final.merge(dataframes[name][columns], on="Month", how="left")

months = pd.date_range("2020-01", "2024-12", freq="MS").strftime("%Y-%m")
full_index = pd.MultiIndex.from_product([sorted(specified_states), months], names=["State", "Month"])
final = final.set_index(["State", "Month"]).reindex(full_index).reset_index()

expected_rows = 13 * 60
if final.shape[0] != expected_rows:
    raise ValueError(f"Expected {expected_rows} rows, got {final.shape[0]}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(folder_path, f"Final_Merged_Soybean_Dataset_{timestamp}.xlsx")
final.to_excel(output_file, index=False)

print("Final dataset shape:", final.shape)
print("Saved at:", output_file)
