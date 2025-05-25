import os
import glob
from loguru import logger
import numpy as np
import pandas as pd
from data_io.global_data import GlobalData

def main():
    global_data = GlobalData("C:/WorkingData/Documents/2_Coding/Python/aaoca_ivus_analysis/data_eacvi")
    glob_df = global_data.create_global_df()

    print(glob_df)

if __name__ == "__main__":
    main()