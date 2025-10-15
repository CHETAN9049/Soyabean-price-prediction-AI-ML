import pandas as pd

input_file = "RS_Session_267_AU_2438_D.i.csv"
msp_df = pd.read_csv(input_file, encoding="utf-8-sig")
msp_df.columns = msp_df.columns.str.strip()

msp_df["Date"] = msp_df["Year"].astype(str) + "-10"
soyabean_msp = dict(zip(msp_df["Date"], msp_df["Soybean"]))
rapeseed_msp = dict(zip(msp_df["Date"], msp_df["Rapeseed"]))

monthly_dates = pd.date_range(start="2020-01-01", end="2025-07-01", freq="MS")
monthly_df = pd.DataFrame({"Date": monthly_dates.strftime("%Y-%m")})

def fill_monthly_msp(date_list, msp_dict):
    filled = []
    sorted_keys = sorted(msp_dict.keys())
    for date in date_list:
        applicable_msp = None
        for key in sorted_keys:
            if date >= key:
                applicable_msp = msp_dict[key]
            else:
                break
        filled.append(applicable_msp)
    return filled

monthly_df["Soybean MSP Monthly"] = fill_monthly_msp(monthly_df["Date"], soyabean_msp)
monthly_df["Rapeseed MSP Monthly"] = fill_monthly_msp(monthly_df["Date"], rapeseed_msp)

output_file = "monthly_msp_output.xlsx"
monthly_df.to_excel(output_file, index=False, sheet_name="Monthly MSP")
print(f"Exported monthly MSP data to {output_file}")
