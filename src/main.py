import os
import glob
from loguru import logger
import numpy as np
import pandas as pd
from data_io.global_data import GlobalData
from data_io.patient_data import LoadIndividualData

GLOBAL_PATH = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi"

def main():
    global_data = GlobalData(GLOBAL_PATH)
    glob_df, ids = global_data.create_global_df()

    pat_data = LoadIndividualData(GLOBAL_PATH, 'narco_119')
    pat_data.process_patient_data()


if __name__ == "__main__":
    main()