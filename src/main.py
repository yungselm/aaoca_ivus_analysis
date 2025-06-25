from __future__ import annotations

from data_io.global_data import GlobalData
from stats.global_stats import GlobalStats
from data_io.patient_data import LoadIndividualData
from loguru import logger
from preprocessing.patient_preprocessing import PatientPreprocessing

# GLOBAL_PATH = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi"
# GLOBAL_OUTPUT = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi/output/global_stats"
# PATIENT_OUTPUT = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi/output/patient_stats"
GLOBAL_PATH = "D:/00_coding/aaoca_ivus_analysis/data"
GLOBAL_OUTPUT = "D:/00_coding//aaoca_ivus_analysis/data/output/global_stats"
PATIENT_OUTPUT = "D:/00_coding//aaoca_ivus_analysis/data/output/patient_stats"

def main():
    global_data = GlobalData(GLOBAL_PATH)
    glob_df, ids = global_data.create_global_df()
    logger.info(f"Loaded Global Data")

    global_stats = GlobalStats(glob_df, GLOBAL_OUTPUT)
    # Plot default (rest & dobu) pressure metrics
    global_stats.global_stats()

    # Plot only rest vs dobu for lumen metrics
    global_stats.plot_global_change(mode="lumen")
    global_stats.plot_global_change(mode="mln")

    ids = [
        "narco_2",
        "narco_3",
        "narco_4",
        "narco_5",
        "narco_10",
        "narco_12",
        "narco_24",
        "narco_119",
        "narco_122",
        "narco_192",
        "narco_199",
        "narco_200",
        "narco_216",
        "narco_218",
        "narco_234",
        "narco_303",
        "narco_295",
        "narco_281",
        "narco_282",
        "narco_288",
        "narco_276",
    ]
    for id in ids:
        try:
            # Plot individual patient data
            individual_data = LoadIndividualData(GLOBAL_PATH, id)
            pat_data = individual_data.process_patient_data()
            logger.info(f"Loaded Patient Data for {id}")

            patient_stats = PatientPreprocessing(
                pat_data, PATIENT_OUTPUT, GLOBAL_OUTPUT
            )
            patient_stats.process_case()
        except Exception as e:
            logger.warning(f"Skipping {id} due to error: {e}")


if __name__ == "__main__":
    main()
