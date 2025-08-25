from __future__ import annotations

from stats.patient_stats import PatientStats


GLOBAL_PATH = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data"
GLOBAL_OUTPUT = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data/output/global_stats"
PATIENT_OUTPUT = "C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data/output/patient_stats"
# GLOBAL_PATH = "D:/00_coding/aaoca_ivus_analysis/data"
# GLOBAL_OUTPUT = "D:/00_coding//aaoca_ivus_analysis/data/output/global_stats"
# PATIENT_OUTPUT = "D:/00_coding//aaoca_ivus_analysis/data/output/patient_stats"


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

    # ids = [
    #     "narco_2",
    #     "narco_3",
    #     # "narco_4",
    #     "narco_5",
    #     "narco_10",
    #     "narco_12",
    #     "narco_24",
    #     "narco_119",
    #     "narco_122",
    #     "narco_192",
    #     "narco_199",
    #     "narco_200",
    #     "narco_208",
    #     "narco_210",
    #     "narco_216",
    #     "narco_218",
    #     "narco_219",
    #     "narco_234",
    #     "narco_241",
    #     "narco_247",
    #     "narco_248",
    #     "narco_249",
    #     "narco_250",
    #     "narco_264",
    #     "narco_267",
    #     "narco_259",
    #     "narco_276",
    #     "narco_279",
    #     "narco_281",
    #     "narco_282",
    #     "narco_288",
    #     "narco_295",
    #     "narco_303",
    #     "narco_306",
    #     "narco_307",
    #     # "narco_324",
    # ]
    # for id in ids:
    #     try:
    #         # Plot individual patient data
    #         individual_data = LoadIndividualData(GLOBAL_PATH, id)
    #         pat_data = individual_data.process_patient_data()
    #         logger.info(f"Loaded Patient Data for {id}")

    #         patient_stats = PatientPreprocessing(
    #             pat_data, PATIENT_OUTPUT, GLOBAL_OUTPUT
    #         )
    #         patient_stats.process_case(plot=False)
    #     except Exception as e:
    #         logger.warning(f"Skipping {id} due to error: {e}")

    patient_stats = PatientStats(GLOBAL_OUTPUT, GLOBAL_OUTPUT)
    patient_stats.run()


if __name__ == "__main__":
    main()
