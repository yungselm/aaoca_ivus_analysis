""" Loads global pressure and IVUS measurements"""
import os
import pandas as pd
import numpy as np

from typing import List

class GlobalData:
    def __init__(self, path):
        self.working_dir = path
        self.df_all: pd.DataFrame = pd.DataFrame()

    
    def create_global_df(self) -> pd.DataFrame:
        pressure_df = self.load_pressure_data()
        ivus_df = self.load_ivus_data()

        df = pd.merge(pressure_df, ivus_df, left_on='patient_id', right_on='narco_id')
        df.drop(columns=['narco_id'])

        self.df_all = df
        ids = self.collect_ids()
        
        return df, ids
    
    def collect_ids(self) -> List:
        return self.df_all["patient_id"].unique()

    def load_pressure_data(self) -> pd.DataFrame:
        dir = os.path.join(self.working_dir, "results.xlsx")
        df = pd.read_excel(dir)

        return df

    def _clean_pressure_data(self):
        pass

    def load_ivus_data(self):
        dir = os.path.join(self.working_dir, "IVUS_data.xlsx")
        df = pd.read_excel(dir)
        df = self._clean_ivus_data(df)

        return df

    @staticmethod
    def _clean_ivus_data(df) -> pd.DataFrame:
        df['NARCO_ID'] = df['NARCO_ID'].astype(str).apply(lambda x: 'narco_' + x)
        df = df.iloc[:, :-3]
        # all col names to lower
        df.columns = [col.lower() for col in df.columns]

        # calculate percent stenosis for all by using biggest reference per column
        reference_cols = ["reference_a_rest", "reference_a_stress", "reference_a_adenosine"]

        df["max_reference"] = df[reference_cols].max(axis=1)

        df["ostial_rest_mln"] = round((1 - (df["ostial_a_rest"] / df["max_reference"])) * 100, 0)
        df["mla_rest_mln"] = round((1 - (df["mla_rest"] / df["max_reference"])) * 100, 0)
        df["ostial_stress_mln"] = round((1 - (df["ostial_a_stress"] / df["max_reference"])) * 100, 0)
        df["mla_stress_mln"] = round((1 - (df["mla_stress"] / df["max_reference"])) * 100, 0)
        df["ostial_adenosine_mln"] = round((1 - (df["ostial_a_adenosine"] / df["max_reference"])) * 100, 0)
        df["mla_adenosine_mln"] = round((1 - (df["mla_adenosine"] / df["max_reference"])) * 100, 0)

        df = df.drop(columns=["max_reference"])

        return df
