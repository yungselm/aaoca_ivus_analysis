from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
import scipy.stats as stats

class PatientStats:
    def __init__(self, input_dir: str, output_path: str):
        self.input_dir = input_dir
        self.output_path = output_path
        self.data = self.load_patient_data()

    def load_patient_data(self) -> pd.DataFrame:
        """
        Load patient data from input path.
        """
        try:
            path = os.path.join(self.input_dir, "local_patient_stats.csv")
            data = pd.read_csv(path)
            logger.info(f"Loaded patient data from {self.input_dir}")
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")

        self.data = data

    def process_local_patient_data(self) -> pd.DataFrame:
        pass

    def exploratory_data_analysis(self):
        pass