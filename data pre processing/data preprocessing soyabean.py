import pandas as pd
import calendar
import os

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists('Soyabean_Prices_Normalized.xlsx'))

file_path = os.path.join(os.getcwd(), 'Soyabean_Prices_Normalized.xlsx')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}. Please check the file path or name.")

df = pd.read_excel(file_path)

month_map = {month: f'{index:02d}' for index, month in enumerate(calendar.month_name) if month}
df['Month'] = df['Month'].str.capitalize().map(month_map)
df['Date'] = df['Year'].astype(str) + '-' + df['Month']
df = df.drop(columns=['Month', 'Year'])

def impute_missing_prices(df, column='Price'):
    for date in df['Date'].unique():
        date_mask = df['Date'] == date
        median_value = df[date_mask][column].median()
        df.loc[date_mask & df[column].isna(), column] = median_value
    return df

df = impute_missing_prices(df, 'Price')

df = df.drop_duplicates()

all_dates = [f"{year}-{month:02d}" for year in range(2020, 2025) for month in range(1, 13)]
all_states = df['State'].unique()
print(f"Unique months: {len(all_dates)}")
print(f"Unique states: {len(all_states)}")
print(f"Expected rows before imputation: {len(all_states) * len(all_dates)}")

all_combinations = pd.MultiIndex.from_product([all_dates, all_states], names=['Date', 'State'])
full_df = pd.DataFrame(index=all_combinations).reset_index()

df = full_df.merge(df, on=['Date', 'State'], how='left')

for col in ['Price', 'PrevMonthPrice', 'PrevYearPrice']:
    df = impute_missing_prices(df, col)

calc_change_prev_month = (df['Price'] - df['PrevMonthPrice']) / df['PrevMonthPrice']
calc_change_prev_year = (df['Price'] - df['PrevYearPrice']) / df['PrevYearPrice']

df['ChangePrevMonth'] = df['ChangePrevMonth'].fillna(calc_change_prev_month)
df['ChangePrevYear']   = df['ChangePrevYear'].fillna(calc_change_prev_year)

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df = df.sort_values(['Date', 'State']).reset_index(drop=True)

print(f"Final dataset shape: {df.shape}")
print(f"Unique months after processing: {df['Date'].nunique()}")
print(f"Unique states after processing: {df['State'].nunique()}")

df.to_csv('Soyabean_Prices_Cleaned_Origin.csv', index=False)

print(df.head())
