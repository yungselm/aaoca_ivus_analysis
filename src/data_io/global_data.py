"""Loads global pressure and IVUS measurements"""

from __future__ import annotations

import os
from typing import List

import pandas as pd


class GlobalData:
    def __init__(self, path):
        self.working_dir = path
        self.df_all: pd.DataFrame = pd.DataFrame()

    def create_global_df(self) -> pd.DataFrame:
        pressure_df = self.load_pressure_data()
        ivus_df = self.load_ivus_data()

        df = pd.merge(pressure_df, ivus_df, left_on="patient_id", right_on="narco_id")
        df.drop(columns=["narco_id"])

        self.df_all = df
        self._clean_pressure_data()
        ids = self.collect_ids()

        return self.df_all, ids

    def collect_ids(self) -> List:
        return self.df_all["patient_id"].unique()

    def load_pressure_data(self) -> pd.DataFrame:
        dir = os.path.join(self.working_dir, "results.xlsx")
        df = pd.read_excel(dir)

        return df

    def _clean_pressure_data(self):
        """Remove columns ending with '_high', '_low', or containing '_low' anywhere in the name."""
        cols_to_drop = [
            col for col in self.df_all.columns if col.endswith("_high") or "_low" in col
        ]
        self.df_all = self.df_all.drop(columns=cols_to_drop)

    def load_ivus_data(self):
        dir = os.path.join(self.working_dir, "IVUS_data.xlsx")
        df = pd.read_excel(dir)
        df = self._clean_ivus_data(df)

        return df

    @staticmethod
    def _clean_ivus_data(df) -> pd.DataFrame:
        df["NARCO_ID"] = df["NARCO_ID"].astype(str).apply(lambda x: "narco_" + x)
        df = df.iloc[:, :-3]
        # all col names to lower
        df.columns = [col.lower() for col in df.columns]

        # calculate percent stenosis for all by using biggest reference per column
        reference_cols = [
            "reference_a_rest",
            "reference_a_stress",
            "reference_a_adenosine",
        ]

        df["max_reference"] = df[reference_cols].max(axis=1)

        df["ostial_rest_mln"] = round(
            (1 - (df["ostial_a_rest"] / df["max_reference"])) * 100, 0
        )
        df["mla_rest_mln"] = round(
            (1 - (df["mla_rest"] / df["max_reference"])) * 100, 0
        )
        df["ostial_stress_mln"] = round(
            (1 - (df["ostial_a_stress"] / df["max_reference"])) * 100, 0
        )
        df["mla_stress_mln"] = round(
            (1 - (df["mla_stress"] / df["max_reference"])) * 100, 0
        )
        df["ostial_adenosine_mln"] = round(
            (1 - (df["ostial_a_adenosine"] / df["max_reference"])) * 100, 0
        )
        df["mla_adenosine_mln"] = round(
            (1 - (df["mla_adenosine"] / df["max_reference"])) * 100, 0
        )

        df = df.drop(columns=["max_reference"])

        return df
