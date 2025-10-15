import pandas as pd
import os

output_dir = r'C:\Users\cheta\Downloads\original dataset'
os.makedirs(output_dir, exist_ok=True)  
input_file = os.path.join(output_dir, 'Soyabean_Arrivals_Combined.xlsx')
output_file = os.path.join(output_dir, 'Soyabean_Arrivals_Cleaned_Origin.csv')

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists(input_file))
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"File not found at: {input_file}. "
        f"Please ensure 'Soyabean_Arrivals_Combined.xlsx' is in {output_dir} "
        f"or update the input_file path in the script."
    )

df = pd.read_excel(input_file)

soybean_states = [
    'Andhra Pradesh', 'Chattisgarh', 'Gujarat', 'Karnataka', 'Madhya Pradesh', 'Maharashtra',
    'Manipur', 'Nagaland', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Uttrakhand'
]

df = df[df['State'].isin(soybean_states)]

month_map = {str(index): f'{index:02d}' for index in range(1, 13)}
df['Month'] = df['Month'].astype(str).map(month_map)
df['Date'] = df['Year'].astype(str) + '-' + df['Month']
df = df.drop(columns=['Month', 'Year'])

df['Market Arrivals (given year)'] = df['Market Arrivals (given year)'].fillna(0)
df['Market Arrivals (previous year)'] = df['Market Arrivals (previous year)'].fillna(0)

df = df.drop_duplicates()

all_dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS').strftime('%Y-%m').tolist()
all_states = sorted(soybean_states)
print(f"Unique months: {len(all_dates)}")
print(f"Unique states: {len(all_states)}")
print(f"Expected rows before imputation: {len(all_states) * len(all_dates)}")

all_combinations = pd.MultiIndex.from_product([all_dates, all_states], names=['Date', 'State'])
full_df = pd.DataFrame(index=all_combinations).reset_index()
df = full_df.merge(df, on=['Date', 'State'], how='left')

df['Market Arrivals (given year)'] = df['Market Arrivals (given year)'].fillna(0)
df['Market Arrivals (previous year)'] = df['Market Arrivals (previous year)'].fillna(0)

df['% (+/-) WRT(previous year)'] = (
    ((df['Market Arrivals (given year)'] - df['Market Arrivals (previous year)']) /
     df['Market Arrivals (previous year)'].replace(0, 1)) * 100
).round(2)

df['State%'] = df.groupby('Date')['Market Arrivals (given year)'].transform(
    lambda x: (x / x.sum() * 100).round(2) if x.sum() != 0 else 0
)

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

df = df.sort_values(['Date', 'State']).reset_index(drop=True)

print(f"Final dataset shape: {df.shape}")
print(f"Unique months after processing: {df['Date'].nunique()}")
print(f"Unique states after processing: {df['State'].nunique()}")

df.to_csv(output_file, index=False)

print(df.head())