

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_FILE = "manufacturing_production_uncleaned.csv"
OUTPUT_FILE = "manufacturing_production_cleaned.csv"
REPORT_FILE = "data_cleaning_report.txt"

report_lines = []


def log(msg):
    """Collect a line for the cleaning report and also print it."""
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv('manufacturing_production_uncleaned.csv')
log(f"Loaded raw file: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. DROP COMPLETELY BLANK ROWS
#    A handful of rows only contain a Record_ID with every other field empty.
#    These carry no usable information, so they are removed.
# ---------------------------------------------------------------------------
before = len(df)
df = df[df["Order_ID"].notna()].copy()
log(f"Dropped {before - len(df)} completely blank rows (kept Record_ID only)")

# ---------------------------------------------------------------------------
# 3. STRIP WHITESPACE FROM EVERY TEXT COLUMN
#    Many fields (e.g. "  Plant B  ", "  N. Fernandes  ") have stray
#    leading/trailing spaces that break grouping in Power BI.
# ---------------------------------------------------------------------------
text_cols = df.select_dtypes(include="object").columns.tolist()
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"nan": np.nan, "": np.nan})
log(f"Trimmed whitespace on {len(text_cols)} text columns")

# ---------------------------------------------------------------------------
# 4. STANDARDISE Plant_Location
#    Raw values mix casing and separators: "PLANT A", "plant b", "Plant-D".
#    Normalise all of them to "Plant A" ... "Plant E".
# ---------------------------------------------------------------------------
def clean_plant(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"plant[\s\-]*([A-E])", val, re.IGNORECASE)
    return f"Plant {match.group(1).upper()}" if match else val

df["Plant_Location"] = df["Plant_Location"].apply(clean_plant)

# ---------------------------------------------------------------------------
# 5. STANDARDISE Shift
#    Raw data mixes two naming schemes for the same 3 real shifts:
#        Morning / Evening / Night   <-- primary scheme
#        Day, 1st Shift, 2nd Shift   <-- alternate scheme
#    ASSUMPTION (documented for the project write-up):
#        "Day"        -> Morning
#        "1st Shift"  -> Morning
#        "2nd Shift"  -> Evening
#    This mapping was chosen to match standard 2/3-shift factory patterns.
#    Adjust SHIFT_MAP below if your factory uses a different convention.
# ---------------------------------------------------------------------------
SHIFT_MAP = {
    "morning": "Morning",
    "evening": "Evening",
    "night": "Night",
    "day": "Morning",
    "1st shift": "Morning",
    "2nd shift": "Evening",
}
df["Shift"] = df["Shift"].str.lower().map(SHIFT_MAP)

# ---------------------------------------------------------------------------
# 6. STANDARDISE Quality_Grade
# ---------------------------------------------------------------------------
GRADE_MAP = {
    "grade a": "A", "a": "A",
    "grade b": "B", "b": "B",
    "c": "C",
    "a+": "A+",
    "reject": "Reject",
}
df["Quality_Grade"] = df["Quality_Grade"].str.lower().map(GRADE_MAP)

# ---------------------------------------------------------------------------
# 7. STANDARDISE Inspection_Passed -> Yes / No
# ---------------------------------------------------------------------------
INSPECTION_MAP = {"y": "Yes", "yes": "Yes", "n": "No", "no": "No"}
df["Inspection_Passed"] = df["Inspection_Passed"].str.lower().map(INSPECTION_MAP)

# ---------------------------------------------------------------------------
# 8. STANDARDISE Maintenance_Status
# ---------------------------------------------------------------------------
MAINT_MAP = {"ok": "OK", "scheduled": "Scheduled", "due": "Due", "overdue": "Overdue"}
df["Maintenance_Status"] = df["Maintenance_Status"].str.lower().map(MAINT_MAP)

# ---------------------------------------------------------------------------
# 9. STANDARDISE Raw_Material casing
# ---------------------------------------------------------------------------
MATERIAL_MAP = {
    "steel": "Steel", "aluminum": "Aluminum", "cast iron": "Cast Iron",
    "abs plastic": "ABS Plastic", "copper": "Copper",
    "polycarbonate": "Polycarbonate",
}
df["Raw_Material"] = df["Raw_Material"].str.lower().map(MATERIAL_MAP)

# ---------------------------------------------------------------------------
# 10. STANDARDISE Supplier_Name
#     "SteelCorp Ltd" and "Steelcorp Ltd." are the same supplier written
#     inconsistently -> merge into one canonical spelling.
# ---------------------------------------------------------------------------
SUPPLIER_MAP = {
    "steelcorp ltd": "SteelCorp Ltd",
    "steelcorp ltd.": "SteelCorp Ltd",
    "global alloys": "Global Alloys",
    "metalworks pvt ltd": "MetalWorks Pvt Ltd",
    "poly materials inc": "Poly Materials Inc",
    "precision components co": "Precision Components Co",
}
df["Supplier_Name"] = df["Supplier_Name"].str.lower().map(SUPPLIER_MAP)

# ---------------------------------------------------------------------------
# 11. CLEAN Supervisor_Name
#     - "unknown" (any case) is treated as missing, not a real name.
#     - Reformat "Last, F." style entries (e.g. "D'Souza, L.") into the
#       same "F. Last" style used everywhere else.
# ---------------------------------------------------------------------------
df.loc[df["Supervisor_Name"].str.lower() == "unknown", "Supervisor_Name"] = np.nan

def reformat_name(val):
    if pd.isna(val):
        return val
    match = re.match(r"^(.+),\s*([A-Za-z])\.?$", val)
    if match:
        last, initial = match.group(1), match.group(2)
        return f"{initial}. {last}"
    return val

df["Supervisor_Name"] = df["Supervisor_Name"].apply(reformat_name)

# ---------------------------------------------------------------------------
# 12. PARSE Production_Date
#     The source file mixes several date formats. Each is identified by
#     its separator / pattern and parsed explicitly (safer than letting
#     pandas guess, which mis-reads ambiguous dates like 07-21-2026).
#         YYYY-MM-DD            -> ISO format
#         YYYY/MM/DD HH:MM      -> ISO with time
#         DD/MM/YYYY            -> slash, international format
#         MM-DD-YYYY            -> dash, US format (no month name)
#         DD-Mon-YYYY           -> dash with month name, e.g. 15-Aug-2026
# ---------------------------------------------------------------------------
def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    val = val.strip()
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
            return pd.to_datetime(val, format="%Y-%m-%d")
        if re.match(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$", val):
            return pd.to_datetime(val, format="%Y/%m/%d %H:%M")
        if re.match(r"^\d{2}/\d{2}/\d{4}$", val):
            return pd.to_datetime(val, format="%d/%m/%Y")
        if re.match(r"^\d{2}-[A-Za-z]{3}-\d{4}$", val):
            return pd.to_datetime(val, format="%d-%b-%Y")
        if re.match(r"^\d{2}-\d{2}-\d{4}$", val):
            return pd.to_datetime(val, format="%m-%d-%Y")
    except ValueError:
        return pd.NaT
    return pd.NaT  # any pattern not recognised

df["Production_Date"] = df["Production_Date"].apply(parse_date)
unparsed = df["Production_Date"].isna().sum()
log(f"Parsed Production_Date across 5 mixed input formats ({unparsed} could not be parsed)")

# ---------------------------------------------------------------------------
# 13. FIX NUMERIC DATA ERRORS
# ---------------------------------------------------------------------------
# 13a. Actual_Quantity_Produced: negative values are sign-entry errors
#      (a quantity produced cannot be negative) -> take absolute value.
neg_qty = (df["Actual_Quantity_Produced"] < 0).sum()
df["Actual_Quantity_Produced"] = df["Actual_Quantity_Produced"].abs()
log(f"Fixed {neg_qty} negative Actual_Quantity_Produced values (converted to absolute value)")

# 13b. Cycle_Time_Sec: same issue, time cannot be negative.
neg_cycle = (df["Cycle_Time_Sec"] < 0).sum()
df["Cycle_Time_Sec"] = df["Cycle_Time_Sec"].abs()
log(f"Fixed {neg_cycle} negative Cycle_Time_Sec values (converted to absolute value)")

# 13c. Defect_Count: 99999 is a sentinel/error value, not a real count.
#      Treat it as missing so it can be imputed like other genuine gaps.
sentinel = (df["Defect_Count"] == 99999).sum()
df.loc[df["Defect_Count"] == 99999, "Defect_Count"] = np.nan
log(f"Flagged {sentinel} Defect_Count values of 99999 as data-entry errors -> set to missing")

# 13d. OEE_Percent: Overall Equipment Effectiveness is a percentage and is
#      physically capped at 100. Values above 100 are measurement/rounding
#      errors -> capped at 100.
capped = (df["OEE_Percent"] > 100).sum()
df["OEE_Percent"] = df["OEE_Percent"].clip(upper=100)
log(f"Capped {capped} OEE_Percent values that exceeded the physical maximum of 100%")

# ---------------------------------------------------------------------------
# 14. IMPUTE REMAINING MISSING NUMERIC VALUES
#     Use the median of the same Product_Line as a more representative
#     fill than a single global median.
# ---------------------------------------------------------------------------
numeric_impute_cols = ["Batch_Size", "Actual_Quantity_Produced", "Defect_Count",
                        "Energy_Consumption_kWh", "Scrap_Cost_USD"]

for col in numeric_impute_cols:
    missing_before = df[col].isna().sum()
    df[col] = df.groupby("Product_Line")[col].transform(
        lambda s: s.fillna(s.median())
    )
    df[col] = df[col].fillna(df[col].median())  # fallback for any edge case
    log(f"Imputed {missing_before} missing values in {col} using Product_Line median")

# ---------------------------------------------------------------------------
# 15. FILL MISSING CATEGORICAL VALUES
#     These fields are legitimately optional in the source system
#     (e.g. inspection isn't logged for every batch), so rows are kept
#     and missing values are labelled instead of dropping ~25% of the data.
# ---------------------------------------------------------------------------
categorical_fill = {
    "Plant_Location": "Unknown",
    "Shift": "Unknown",
    "Operator_ID": "Unknown",
    "Supervisor_Name": "Unknown",
    "Raw_Material": "Unknown",
    "Supplier_Name": "Unknown",
    "Quality_Grade": "Not Graded",
    "Inspection_Passed": "Not Recorded",
    "Maintenance_Status": "Not Scheduled",
    "Remarks": "No Remarks",
}
for col, fill_val in categorical_fill.items():
    missing_before = df[col].isna().sum()
    df[col] = df[col].fillna(fill_val)
    log(f"Filled {missing_before} missing values in {col} with '{fill_val}'")

# ---------------------------------------------------------------------------
# 16. REMOVE EXACT DUPLICATE ROWS (safety net, none expected)
# ---------------------------------------------------------------------------
dupes = df.duplicated().sum()
df = df.drop_duplicates()
log(f"Removed {dupes} exact duplicate rows")

# ---------------------------------------------------------------------------
# 17. ADD DERIVED KPI COLUMNS FOR POWER BI
# ---------------------------------------------------------------------------
df["Yield_Percent"] = (df["Actual_Quantity_Produced"] / df["Planned_Quantity"] * 100).round(2)
df["Defect_Rate_Percent"] = np.where(
    df["Actual_Quantity_Produced"] > 0,
    (df["Defect_Count"] / df["Actual_Quantity_Produced"] * 100).round(2),
    0,
)
log("Added derived columns: Yield_Percent, Defect_Rate_Percent")

# ---------------------------------------------------------------------------
# 18. FINAL TIDY-UP: round floats, fix dtypes
# ---------------------------------------------------------------------------
round_2 = ["Machine_Age_Years", "Downtime_Minutes", "Cycle_Time_Sec", "Temperature_C",
           "Humidity_Pct", "Vibration_mm_s", "Energy_Consumption_kWh", "Unit_Cost_USD",
           "Scrap_Cost_USD", "Labor_Hours", "OEE_Percent"]
for col in round_2:
    df[col] = df[col].round(2)

int_cols = ["Batch_Size", "Planned_Quantity", "Actual_Quantity_Produced",
            "Defect_Count", "Last_Maintenance_Days_Ago"]
for col in int_cols:
    df[col] = df[col].round(0).astype("Int64")

df["Production_Date"] = df["Production_Date"].dt.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# 19. SAVE OUTPUT
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
log(f"\nFinal cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
log(f"Saved cleaned file to {OUTPUT_FILE}")

with open(REPORT_FILE, "w") as f:
    f.write("\n".join(report_lines))
