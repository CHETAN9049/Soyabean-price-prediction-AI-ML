import pandas as pd
import numpy as np
import calendar
import os

output_dir = r'C:\Users\cheta\Downloads\original dataset'
os.makedirs(output_dir, exist_ok=True)  
input_file = os.path.join(output_dir, 'Mustard_Prices_Normalized.xlsx')
output_file = os.path.join(output_dir, 'Mustard_Prices_Cleaned_Origin.csv')

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists(input_file))
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"File not found at: {input_file}. "
        f"Please ensure 'Mustard_Prices_Normalized.xlsx' is in {output_dir} "
        f"or update the input_file path in the script."
    )

df = pd.read_excel(input_file)

soybean_states = [
    'Andhra Pradesh', 'Chattisgarh', 'Gujarat', 'Karnataka', 'Madhya Pradesh', 'Maharashtra',
    'Manipur', 'Nagaland', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Uttrakhand'
]

df = df[df['State'].isin(soybean_states)]

month_map = {month: f'{index:02d}' for index, month in enumerate(calendar.month_name) if month}
df['Month'] = df['Month'].str.capitalize().map(month_map)
df['Date'] = df['Year'].astype(str) + '-' + df['Month']
df = df.drop(columns=['Month', 'Year'])

def impute_missing_prices(df, column='Price'):
    for date in df['Date'].unique():
        date_mask = df['Date'] == date
        median_value = df[date_mask][column].median()
        if pd.isna(median_value):  
            median_value = df[column].median()
        df.loc[date_mask & df[column].isna(), column] = median_value
    return df

df = impute_missing_prices(df, 'Price')

df = df.drop_duplicates()


all_dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS').strftime('%Y-%m').tolist()
all_states = sorted(soybean_states)

all_combinations = pd.MultiIndex.from_product([all_dates, all_states], names=['Date', 'State'])
full_df = pd.DataFrame(index=all_combinations).reset_index()

df = full_df.merge(df, on=['Date', 'State'], how='left')

df = impute_missing_prices(df, 'Price')
df = impute_missing_prices(df, 'PrevMonthPrice')
df = impute_missing_prices(df, 'PrevYearPrice')

df['ChangePrevMonth'] = np.where(
    df['PrevMonthPrice'] > 0,
    ((df['Price'] - df['PrevMonthPrice']) / df['PrevMonthPrice']) * 100,
    np.nan
)

df['ChangePrevYear'] = np.where(
    df['PrevYearPrice'] > 0,
    ((df['Price'] - df['PrevYearPrice']) / df['PrevYearPrice']) * 100,
    np.nan
)

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

df = df.sort_values(['Date', 'State']).reset_index(drop=True)

print(f"Final dataset shape: {df.shape}")
print(f"Unique months after processing: {df['Date'].nunique()}")
print(f"Unique states after processing: {df['State'].nunique()}")

df.to_csv(output_file, index=False)

print(df.head())
