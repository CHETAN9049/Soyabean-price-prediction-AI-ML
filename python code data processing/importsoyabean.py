import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

file_path = "importsoyabean.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", skiprows=6, usecols=[0, 3], names=['Year', 'Soyabean'])

df = df.dropna(subset=['Year', 'Soyabean'])
df = df[df['Year'].str.match(r'\d{4}-\d{4}')]
data = dict(zip(df['Year'], df['Soyabean'].astype(int)))

fiscal_years = {
    '2019-2020': ('2019-11', '2020-10'),
    '2020-2021': ('2020-11', '2021-10'),
    '2021-2022': ('2021-11', '2022-10'),
    '2022-2023': ('2022-11', '2023-10'),
    '2023-2024': ('2023-11', '2024-10'),
    '2024-2025': ('2024-11', '2025-07')  
}

monthly_data = []
for fy, (start_str, end_str) in fiscal_years.items():
    start_date = datetime.strptime(start_str, '%Y-%m')
    end_date = datetime.strptime(end_str, '%Y-%m')

    if fy in data:
        total = data[fy]
    else:
        total = list(data.values())[-1]  

    months_list = []
    current = start_date
    while current <= end_date:
        months_list.append(current)
        current += relativedelta(months=+1)

    monthly_value = round(total / len(months_list)) if len(months_list) > 0 else 0

    for m in months_list:
        monthly_data.append({
            'Date': m.strftime('%Y-%m'),
            'Soyabean Oil Monthly': monthly_value
        })

output_df = pd.DataFrame(monthly_data).drop_duplicates(subset=['Date']).sort_values('Date')

output_df['Date_dt'] = pd.to_datetime(output_df['Date'], format='%Y-%m')
output_df = output_df[
    (output_df['Date_dt'] >= datetime(2020, 1, 1)) &
    (output_df['Date_dt'] <= datetime(2025, 7, 31))
].drop(columns=['Date_dt'])

output_file = "soyabean_monthly_imports_Jan2020_Jul2025.xlsx"
output_df.to_excel(output_file, index=False, sheet_name="Monthly Imports")
print(f"Results exported to {output_file}")
