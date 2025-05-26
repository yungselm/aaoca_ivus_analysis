import os 
import numpy as np
import pandas as pd
from loguru import logger

from pathlib import Path
from typing import Tuple, List, Dict

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
        self.working_dir: Path = Path(path)
        self.id: str = id
        self.pressure_dir, self.ivus_dir = self.find_dirs()
        self.patient_data = PatientData()

    def process_patient_data(self) -> PatientData:
        self.patient_data.id = self.id
        self.process_pressure()
        self.load_obj_data()
        logger.info(f"Loaded PatientData for {self.id}: {self.patient_data}")
        return self.patient_data
        
    def find_dirs(self) -> Tuple[Path, Path]:
        """
        Finds in the working dir the directories with matching patient id
        in /processed and /3d_ivus. The patient id (e.g., narco_1) can appear in
        any case (upper/lower) and directories may have additional info, e.g.,
        NARCO_1_pressure_eval.
        """
        processed_dir = self.working_dir / "processed"
        ivus_base_dir = self.working_dir / "3d_ivus"

        def find_matching_dir(base_dir: Path) -> Path | None:
            if not base_dir.is_dir():
                return None
            for d in base_dir.iterdir():
                if d.is_dir() and self.id.lower() in d.name.lower():
                    return d
            return None

        pressure_dir = find_matching_dir(processed_dir)
        ivus_dir = find_matching_dir(ivus_base_dir)

        if pressure_dir is None:
            raise FileNotFoundError(f"No processed data directory for {self.id} in {processed_dir}")
        if ivus_dir is None:
            raise FileNotFoundError(f"No IVUS data directory for {self.id} in {ivus_base_dir}")

        return pressure_dir, ivus_dir

    def process_pressure(self) -> None:
        patterns = ['rest_1', 'dobu']
        tuples_pressure: List[Tuple[float, float]] = []
        tuples_time: List[Tuple[float, float]] = []

        for phase in patterns:
            filename = f"{self.id}_pressure_{phase}_average_curve_all.csv"
            filepath = self.pressure_dir / filename

            try:
                df = pd.read_csv(filepath)
                p = df['p_aortic_smooth']
                t = df['time']

                tuples_pressure.append((p.iloc[0], p.max()))
                tuples_time.append((t.iloc[0],  t.iloc[p.idxmax()]))
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                tuples_pressure.append((None, None))
                tuples_time.append((None, None))

        (self.patient_data.pressure_rest,
         self.patient_data.pressure_stress) = tuples_pressure
        (self.patient_data.time_rest,
         self.patient_data.time_stress)     = tuples_time

    def load_obj_data(self) -> None:
        """Loads the .obj files with the 3D aligned IVUS images and assigns to patient_data."""
        states  = ['rest', 'stress']
        objects = ['mesh', 'catheter', 'wall']
        phases  = {'dia': '000', 'sys': '029'}

        for state in states:
            # Case-insensitive state directory search
            candidates = [d for d in self.ivus_dir.iterdir()
                          if d.is_dir() and d.name.lower() == state]
            if not candidates:
                raise FileNotFoundError(f"No '{state}' directory in {self.ivus_dir}")
            state_dir = candidates[0]

            for obj in objects:
                for phase, idx in phases.items():
                    filename = f"{obj}_{idx}_{state}.obj"
                    path = state_dir / filename
                    if not path.exists():
                        raise FileNotFoundError(f"Missing: {path}")

                    arr = self._read_obj_file(path)
                    # map mesh -> contours
                    key_obj = 'contours' if obj == 'mesh' else obj
                    attr = f"{state}_{key_obj}_{phase}"
                    setattr(self.patient_data, attr, arr)

    @staticmethod
    def _read_obj_file(obj_filename: Path) -> np.ndarray:
        obj_points: List[List[float]] = []
        with open(obj_filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    try:
                        x, y, z = map(float, parts[1:4])
                        obj_points.append([x, y, z])
                    except (ValueError, IndexError):
                        continue

        if not obj_points:
            raise ValueError(f"No vertices found in {obj_filename}")

        # Group points by z-value and sort contours from lowest to highest z
        z_groups = {}
        for point in obj_points:
            z = point[2]
            z_groups.setdefault(z, []).append(point)

        # Sort groups by z-value and validate counts
        sorted_z = sorted(z_groups.keys())
        output = []
        for contour_idx, z in enumerate(sorted_z):
            contour_points = z_groups[z]
            
            # Add contour index to each point
            for point in contour_points:
                output.append([contour_idx, *point])  # [contour_idx, x, y, z]

        return np.array(output)