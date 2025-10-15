import pandas as pd
import itertools

selected_states = [
    'Andhra Pradesh', 'Chhattisgarh', 'Gujarat', 'Karnataka', 'Madhya Pradesh',
    'Maharashtra', 'Manipur', 'Nagaland', 'Rajasthan', 'Tamil Nadu', 'Telangana',
    'Uttar Pradesh', 'Uttarakhand'
]
years = [2020, 2021, 2022, 2023, 2024]

all_combinations = pd.DataFrame(
    list(itertools.product(selected_states, years)),
    columns=['States', 'Year']
)

try:
    df = pd.read_excel('cleaned_median.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    print("Error: 'cleaned_median.xlsx' not found in the current directory.")
    exit()

print("Column names in Excel file:", df.columns.tolist())

df.columns = df.columns.str.strip()

expected_columns = {
    "Area (In ' 000 Hectare)": "Area (In '000 Hectare)",
    "Production (In ' 000 Tonne)": "Production (In '000 Tonne)",
    "Yield (In Kg./Hectare)": "Yield (In Kg./Hectare)",
    "States": "States",
    "Year": "Year"
}
df = df.rename(columns=expected_columns)

print("Column names after renaming:", df.columns.tolist())

df = df[df['States'].isin(selected_states)]

try:
    df['Year'] = df['Year'].astype(int)
except ValueError:
    print("Error: 'Year' column contains non-integer values. Please check the data.")
    exit()

df_complete = pd.merge(
    all_combinations,
    df,
    on=['States', 'Year'],
    how='left'
)

df_complete["Area (In '000 Hectare)"] = df_complete["Area (In '000 Hectare)"].fillna(0)
df_complete["Production (In '000 Tonne)"] = df_complete["Production (In '000 Tonne)"].fillna(0)
df_complete["Yield (In Kg./Hectare)"] = df_complete["Yield (In Kg./Hectare)"].fillna(0)

df_complete['Year'] = df_complete['Year'].apply(lambda x: f"{int(x)}-01")

df_complete = df_complete.sort_values(by=['States', 'Year'])

df_complete.to_csv('cleaned_soybean_data_complete.csv', index=False)

print("\nFirst few rows of cleaned data:")
print(df_complete.head())
print(f"Total rows: {len(df_complete)}")