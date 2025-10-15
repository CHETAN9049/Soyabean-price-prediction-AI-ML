import pandas as pd

df = pd.read_excel('merged_rainfall_cleaned.xlsx')

groups = {
    'Andhra Pradesh': ['Coastal Andhra Pradesh', 'Rayalaseema', 'Coastal Andhra Pradesh & Yanam'],
    'Chhattisgarh': ['Chhattisgarh'],
    'Gujarat': ['Gujarat Region', 'Saurashtra & Kutch'],
    'Karnataka': ['North Interior Karnataka'],
    'Madhya Pradesh': ['West Madhya Pradesh', 'East Madhya Pradesh'],
    'Maharashtra': ['Madhya Maharashtra', 'Marathwada', 'Vidarbha'],
    'Manipur': ['Nagaland, Manipur, Mizoram & Tripura'],
    'Nagaland': ['Nagaland, Manipur, Mizoram & Tripura'],
    'Rajasthan': ['West Rajasthan', 'East Rajasthan'],
    'Tamil Nadu': ['Tamil Nadu & Puducherry', 'Tamil Nadu, Puducherry & Karaikal'],
    'Telangana': ['Telangana'],
    'Uttar Pradesh': ['East Uttar Pradesh', 'West Uttar Pradesh'],
    'Uttarakhand': ['Uttarakhand']
}

months_sorted = list(df['Month'].unique())
years_sorted = sorted(df['Year'].unique())

new_rows = []

for year in years_sorted:
    for month in months_sorted:
        month_data = df[(df['Month'] == month) & (df['Year'] == year)]
        for state, subs in groups.items():
            sub_df = month_data[month_data['Sub-Division'].isin(subs)]
            if not sub_df.empty:
                actual_mean = sub_df['Rainfall Actual'].mean()
                normal_mean = sub_df['Rainfall Normal'].mean()
                dep_mean = ((actual_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
            else:
                actual_mean = normal_mean = dep_mean = 0 

            new_rows.append({
                'Month': month,
                'Year': year,
                'State': state,
                'Rainfall Actual (mm)': round(actual_mean, 2),
                'Rainfall Normal (mm)': round(normal_mean, 2),
                'Rainfall Departure (%)': round(dep_mean, 2)
            })

new_df = pd.DataFrame(new_rows)
new_df.to_excel('processed_rainfall.xlsx', index=False)

print(f"Processed dataset saved with {len(new_df)} rows to 'processed_rainfall.xlsx'")
