import os
import glob
from loguru import logger
import numpy as np
import pandas as pd
from data_io.global_data import GlobalData
from data_io.patient_data import LoadIndividualData
from stats.global_stats import GlobalStats
from stats.patient_stats import PatientStats

GLOBAL_PATH = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi"
GLOBAL_OUTPUT = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi/output/global_stats"
PATIENT_OUTPUT = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi/output/patient_stats"

def main():
    # global_data = GlobalData(GLOBAL_PATH)
    # glob_df, ids = global_data.create_global_df()
    # logger.info(f"Loaded Global Data")
    
    # global_stats = GlobalStats(glob_df, GLOBAL_OUTPUT)
    # # Plot default (rest & dobu) pressure metrics
    # global_stats.global_stats()

    # # Plot only rest vs dobu for lumen metrics
    # global_stats.plot_global_change(mode="lumen")
    # global_stats.plot_global_change(mode="mln")

    ids = ['narco_119', 'narco_122', 'narco_216', 'narco_218', 'narco_234']
    for id in ids:
        try:
            # Plot individual patient data
            individual_data = LoadIndividualData(GLOBAL_PATH, id)
            pat_data = individual_data.process_patient_data()
            logger.info(f"Loaded Patient Data for {id}")
            
            patient_stats = PatientStats(pat_data, PATIENT_OUTPUT, GLOBAL_OUTPUT)
            patient_stats.process_case()
        except Exception as e:
            logger.warning(f"Skipping {id} due to error: {e}")

if __name__ == "__main__":
    main()
