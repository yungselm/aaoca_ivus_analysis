import os 
import numpy as np
import pandas as pd

from typing import Tuple

class PatientData:
    def __init__(self, id=None):
        self.id: str = id
        self.rest_contours_dia: np.ndarray = None
        self.rest_catheter_dia: np.ndarray = None
        self.rest_wall_dia: np.ndarray = None
        self.rest_contours_sys: np.ndarray = None
        self.rest_catheter_sys: np.ndarray = None
        self.rest_wall_sys: np.ndarray = None
        self.stress_contours_dia: np.ndarray = None
        self.stress_catheter_dia: np.ndarray = None
        self.stress_wall_dia: np.ndarray = None
        self.stress_contours_sys: np.ndarray = None
        self.stress_catheter_sys: np.ndarray = None
        self.stress_wall_sys: np.ndarray = None
        self.pressure_rest: Tuple[float, float] = (None, None)
        self.time_rest: Tuple[float, float] = (None, None)
        self.pressure_stress: Tuple[float, float] = (None, None)
        self.time_stress: Tuple[float, float] = (None, None)

    def __repr__(self):
        def format_attr(value):
            if isinstance(value, np.ndarray):
                return f"ndarray{value.shape}" if value is not None else "None"
            return value

        info = [f"{key} = {format_attr(value)}" for key, value in vars(self).items()]
        return "PatientData(\n  " + ",\n  ".join(info) + "\n)"

class LoadIndividualData:
    def __init__(self, path, id):
        self.working_dir = path
        self.id: str = id
        self.pressure_dir, self.ivus_dir = self.find_dirs()
        self.patient_data = PatientData()

    def process_patient_data(self) -> PatientData:
        self.process_pressure()
        
    def find_dirs(self):
        """
        Finds in the working dir the directories with matching patient id
        in /processed and /3d_ivus. The patient id (e.g., narco_1) can appear in
        any case (upper/lower) and directories may have additional info, e.g.,
        NARCO_1_pressure_eval.
        """
        processed_dir = os.path.join(self.working_dir, "processed")
        ivus_dir = os.path.join(self.working_dir, "3d_ivus")

        def find_matching_dir(base_dir):
            if not os.path.isdir(base_dir):
                return None
            for d in os.listdir(base_dir):
                if self.id and self.id.lower() in d.lower():
                    return os.path.join(base_dir, d)
            return None

        pressure_dir = find_matching_dir(processed_dir)
        ivus_dir = find_matching_dir(ivus_dir)
        return pressure_dir, ivus_dir

    def process_pressure(self):
        patterns = ['rest_1', 'dobu']
        tuples_pressure = []
        tuples_time = []

        for phase in patterns:
            filename = f"{self.id}_pressure_{phase}_average_curve_all.csv"
            filepath = os.path.join(self.pressure_dir, filename)

            try:
                df = pd.read_csv(filepath)

                pressure = df['p_aortic_smooth']
                time = df['time']

                start_pressure = pressure.iloc[0]
                peak_pressure = pressure.max()
                peak_time = time.iloc[pressure.idxmax()]
                start_time = time.iloc[0]

                tuples_pressure.append((start_pressure, peak_pressure))
                tuples_time.append((start_time, peak_time))

            except Exception as e:
                print(f"Failed to process file: {filepath}\nError: {e}")

        self.patient_data.pressure_rest = tuples_pressure[0]
        self.patient_data.pressure_stress = tuples_pressure[1]
        self.patient_data.time_rest = tuples_time[0]
        self.patient_data.time_stress = tuples_time[1]

        print(self.patient_data)

    @staticmethod
    def _read_obj_file(obj_filename) -> np.array:
        # Read and reduce OBJ file points
        obj_points = []
        with open(obj_filename, 'r') as f:
            for line in f:
                if line.startswith('v '):  # Only process vertex lines
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            obj_points.append([x, y, z])
                        except ValueError:
                            continue

        if not obj_points:
            raise ValueError("No valid vertices found in the OBJ file")

        obj = np.array(obj_points)
        return obj