import pandas as pd
from bs4 import BeautifulSoup

file_mapping = {
    "data (1).html": ("Jan-Apr", "2020"),
    "data (2).html": ("May-Aug", "2020"),
    "data (3).html": ("Sep-Dec", "2020"),
    "data (4).html": ("Jan-Apr", "2021"),
    "data (5).html": ("May-Aug", "2021"),
    "data (6).html": ("Sep-Dec", "2021"),
    "data (7).html": ("Jan-Apr", "2022"),
    "data (8).html": ("May-Aug", "2022"),
    "data (10).html": ("Sep-Dec", "2022"),
    "data (11).html": ("Jan-Apr", "2023"),
    "data (12).html": ("May-Aug", "2023"),
    "data (13).html": ("Sep-Dec", "2023"),
    "data (14).html": ("Jan-Apr", "2024"),
    "data (15).html": ("May-Aug", "2024"),
    "data (16).html": ("Sep-Dec", "2024")
}

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
start_month_map = {
    "Jan-Apr": 0,   
    "May-Aug": 4,   
    "Sep-Dec": 8    
}

def expand_row(row):
    expanded = []
    for cell in row.find_all(["th", "td"]):
        text = cell.get_text(strip=True)
        colspan = int(cell.get("colspan", 1))
        expanded.extend([text] * colspan)
    return expanded

def read_rainfall_file(filepath, year, month_range):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    table = soup.find("table")
    if not table:
        return pd.DataFrame()

    rows = table.find_all("tr")
    if len(rows) < 3:
        return pd.DataFrame()

    header1 = expand_row(rows[0])
    header2 = expand_row(rows[1])
    combined_headers = [f"{h1} {h2}".strip() for h1, h2 in zip(header1, header2)]
    
    data_rows = []
    last_state = None

    for row in rows[2:]:
        cells = row.find_all(["td", "th"])
        values = [cell.get_text(strip=True) for cell in cells]

        if len(values) == len(combined_headers) - 1:
            values.insert(0, last_state)
        else:
            last_state = values[0]

        if len(values) == len(combined_headers):
            data_rows.append(values)

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows, columns=combined_headers)

    start_month_idx = start_month_map[month_range]

    month_data = []
    month_idx = start_month_idx

    for i in range(2, len(df.columns), 3):
        if month_idx >= len(month_names):
            break
        if i + 2 >= len(df.columns):
            break
        temp = df.iloc[:, [1, i, i + 1, i + 2]].copy()  
        temp.columns = ["Sub-Division", "Rainfall Actual", "Rainfall Normal", "Rainfall % Dep"]
        temp["Month"] = month_names[month_idx]
        temp["Year"] = year
        month_data.append(temp)
        month_idx += 1

    return pd.concat(month_data, ignore_index=True) if month_data else pd.DataFrame()

all_data = []
for file, (month_range, year) in file_mapping.items():
    try:
        df = read_rainfall_file(file, year, month_range)
        if not df.empty:
            all_data.append(df)
            print(f"Processed {file}")
        else:
            print(f"No data extracted from {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_excel("merged_rainfall_cleaned.xlsx", index=False)
    print("Final merged file saved as 'merged_rainfall_cleaned.xlsx'")
else:
    print("No data extracted from any file.")
