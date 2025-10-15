import pandas as pd
import os

folder_path = r'C:\Users\cheta\Downloads\export'  

all_data = []

file_year_mapping = {
    "data.html": 2025,
    "data (1).html": 2024,
    "data (2).html": 2023,
    "data (3).html": 2022,
    "data (4).html": 2021,
    "data (5).html": 2020
}

for file_name, year in file_year_mapping.items():
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path):
        try:
            tables = pd.read_html(file_path, header=0)
            df = tables[0]
            print(f"Columns in {file_name}: {df.columns.tolist()}")
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            else:
                df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
            
            expected_columns = ['Month', 'Soyabean Extraction', 'Rapeseed Extraction', 
                              'Groundnut Extraction', 'Rice Bran Extraction', 
                              'Castor Seed Extraction', 'Total']
            if len(df.columns) == len(expected_columns):
                df.columns = expected_columns
            else:
                print(f"Warning: Column count mismatch in {file_name}. Expected {len(expected_columns)}, got {len(df.columns)}")
                continue
            
            df['Year'] = df.apply(
                lambda row: year - 1 if row['Month'].strip() in ['April', 'May', 'June', 'July', 'August', 
                                                                'September', 'October', 'November', 'December'] 
                                else year,
                axis=1
            )
        
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    else:
        print(f"File {file_name} not found")

if all_data:
    merged_data = pd.concat(all_data, ignore_index=True)
    
    numeric_columns = ['Soyabean Extraction', 'Rapeseed Extraction', 'Groundnut Extraction',
                      'Rice Bran Extraction', 'Castor Seed Extraction', 'Total']
    for col in numeric_columns:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
    
    month_order = ['April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December', 'January', 'February', 'March']
    merged_data['Month'] = pd.Categorical(merged_data['Month'], categories=month_order, ordered=True)
    merged_data = merged_data.sort_values(['Year', 'Month'])
    
    output_file = "merged_oilmeal_data_2020_2025.xlsx"
    merged_data.to_excel(output_file, index=False, sheet_name='Oilmeal_Exports')
    
    print(f"Data successfully merged and saved to {output_file}")
else:
    print("No data was processed. Please check the folder path and files.")