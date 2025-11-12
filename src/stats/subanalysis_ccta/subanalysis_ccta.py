import pandas as pd
import numpy as np

data = pd.read_csv("data/L1958Narco_DATA_2025-11-06_1353.csv")
data_bl = data[data["redcap_event_name"] == "baseline_arm_1"]
data_bl = data_bl[data_bl["record_id"] != 1]

data_bl.set_index(["record_id", "redcap_repeat_instrument"], inplace=True)
data_bl = data_bl[data_bl.index.get_level_values("redcap_repeat_instrument").isin([np.nan, "ccta", "invasive_testing", "diagnostic_exams"])]

is_invasive = data_bl.index.get_level_values("redcap_repeat_instrument") == "invasive_testing"
inv_ok = data_bl["inv_protocol___2"].isin([1, "1", True])  # accept numeric 1, string '1' or boolean True

mask = (~is_invasive) | (is_invasive & inv_ok)
data_bl = data_bl[mask]

data_bl = data_bl.groupby("record_id").first()
data_bl = data_bl.dropna(axis=1, how='all')
data_bl = data_bl[data_bl["inv_ffrdobu"].notna() | data_bl["inv_ffrado"].notna()]

# data_ccta = data_bl.filter(like = "ccta_")
data_ccta = data_bl.filter(regex="patient_id|narco_id|age|sex_0_male_1_female|bmi|hx_|ccta_|inv_")
print(data_bl)
print(data_ccta)

data_ccta.to_csv("data/ccta_data.csv")