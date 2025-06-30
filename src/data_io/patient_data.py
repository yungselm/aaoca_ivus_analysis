from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger
import re

class PatientData:
    def __init__(self, id: Optional[str] = None) -> None:
        self.id: Optional[str] = id
        self.rest_contours_dia: Optional[np.ndarray] = None
        self.rest_catheter_dia: Optional[np.ndarray] = None
        self.rest_wall_dia: Optional[np.ndarray] = None
        self.rest_contours_sys: Optional[np.ndarray] = None
        self.rest_catheter_sys: Optional[np.ndarray] = None
        self.rest_wall_sys: Optional[np.ndarray] = None
        self.stress_contours_dia: Optional[np.ndarray] = None
        self.stress_catheter_dia: Optional[np.ndarray] = None
        self.stress_wall_dia: Optional[np.ndarray] = None
        self.stress_contours_sys: Optional[np.ndarray] = None
        self.stress_catheter_sys: Optional[np.ndarray] = None
        self.stress_wall_sys: Optional[np.ndarray] = None
        self.dia_contours_rest: Optional[np.ndarray] = None
        self.dia_catheter_rest: Optional[np.ndarray] = None
        self.dia_wall_rest: Optional[np.ndarray] = None
        self.dia_contours_stress: Optional[np.ndarray] = None
        self.dia_catheter_stress: Optional[np.ndarray] = None
        self.dia_wall_stress: Optional[np.ndarray] = None
        self.sys_contours_rest: Optional[np.ndarray] = None
        self.sys_catheter_rest: Optional[np.ndarray] = None
        self.sys_wall_rest: Optional[np.ndarray] = None
        self.sys_contours_stress: Optional[np.ndarray] = None
        self.sys_catheter_stress: Optional[np.ndarray] = None
        self.sys_wall_stress: Optional[np.ndarray] = None
        self.pressure_rest: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        )  # Currently ignoring missing data by Optional should be handled later
        self.time_rest: Tuple[Optional[float], Optional[float]] = (None, None)
        self.pressure_stress: Tuple[Optional[float], Optional[float]] = (None, None)
        self.time_stress: Tuple[Optional[float], Optional[float]] = (None, None)

    def __repr__(self) -> str:
        def format_attr(value):
            if isinstance(value, np.ndarray):
                return f"ndarray{value.shape}" if value is not None else "None"
            return value

        info = [f"{key} = {format_attr(value)}" for key, value in vars(self).items()]
        return "PatientData(\n  " + ",\n  ".join(info) + "\n)"


class LoadIndividualData:
    def __init__(self, path: Path, id: Optional[str] = None) -> None:
        self.working_dir: Path = Path(path)
        self.id: Optional[str] = id
        self.file_id = id
        self.pressure_dir, self.ivus_dir, self.loess_dir = self.find_dirs()
        self.patient_data = PatientData()

    def process_patient_data(self) -> PatientData:
        self.patient_data.id = self.id
        self.process_pressure()
        self.load_obj_data()
        logger.info(f"Loaded PatientData for {self.id}")
        logger.debug(f"Patient data: {self.patient_data}")
        return self.patient_data

    def find_dirs(self) -> Tuple[Path, Path, Path]:
        """
        Finds in the working dir the directories with matching patient id
        in /processed and /3d_ivus. The patient id (e.g., narco_1) can appear in
        any case (upper/lower) and directories may have additional info, e.g.,
        NARCO_1_pressure_eval.
        """
        processed_base_dir = self.working_dir / "processed"
        ivus_base_dir = self.working_dir / "3d_ivus"
        loess_base_dir = self.working_dir / "ivus"

        def find_matching_dir(base_dir: Path) -> Optional[Path]:
            if not base_dir.is_dir():
                return None
            for d in base_dir.iterdir():
                if d.is_dir() and self.id.lower() in d.name.lower():
                    return d
            return None

        pressure_dir: Optional[Path] = find_matching_dir(processed_base_dir)
        ivus_dir: Optional[Path] = find_matching_dir(ivus_base_dir)
        loess_dir: Optional[Path] = find_matching_dir(loess_base_dir)

        if pressure_dir is None:
            raise FileNotFoundError(
                f"No processed data directory for {self.id} in {processed_base_dir}"
            )
        m = re.search(r'(narco_[0-9]+)', pressure_dir.name, flags=re.IGNORECASE)
        if m:
            self.file_id = m.group(1).lower()   # e.g. "narco_303"
        else:
            self.file_id = self.id 

        if ivus_dir is None:
            raise FileNotFoundError(
                f"No IVUS data directory for {self.id} in {ivus_base_dir}"
            )
        if loess_dir is None:
            raise FileNotFoundError(
                f"No Loess data directory for {self.id} in {loess_dir}"
            )

        return pressure_dir, ivus_dir, loess_dir

    def process_pressure(self) -> None:
        patterns = ["rest_1", "dobu"]
        tuples_pressure: List[Tuple[Optional[float], Optional[float]]] = []
        tuples_time: List[Tuple[Optional[float], Optional[float]]] = []

        for phase in patterns:
            filename = f"{self.file_id}_pressure_{phase}_average_curve_all.csv"
            filepath = self.pressure_dir / filename

            try:
                df = pd.read_csv(filepath)
                p = df["p_aortic_smooth"]
                t = df["time"]

                tuples_pressure.append((p.iloc[0], p.max()))
                tuples_time.append((t.iloc[0], t.iloc[p.idxmax()]))
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                tuples_pressure.append((None, None))
                tuples_time.append((None, None))

        (self.patient_data.pressure_rest, self.patient_data.pressure_stress) = (
            tuples_pressure
        )
        (self.patient_data.time_rest, self.patient_data.time_stress) = tuples_time

    def load_obj_data(self) -> None:
        """Loads the .obj files with the 3D aligned IVUS images and assigns to patient_data."""
        states = ["rest", "stress"]
        objects = ["mesh", "catheter", "wall"]
        phases = {"dia": "000", "sys": "029"}

        for state in states:
            # Case-insensitive state directory search
            candidates = [
                d
                for d in self.ivus_dir.iterdir()
                if d.is_dir() and d.name.lower() == state
            ]
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
                    key_obj = "contours" if obj == "mesh" else obj
                    attr = f"{state}_{key_obj}_{phase}"
                    setattr(self.patient_data, attr, arr)

        # Load comparison states
        comp_folders = {
            "diastole_comparison": ("dia", "diastolic"),
            "systole_comparison": ("sys",  "systolic"),
        }

        for comp_dir, (phase_key, phase_label) in comp_folders.items():
            base = self.ivus_dir / comp_dir
            if not base.is_dir():
                raise FileNotFoundError(f"Missing comparison folder: {base}")

            for obj in objects:
                for phase_suffix, idx in phases.items():  # phase_suffix is "dia"/"sys", idx "000"/"029"
                    filename = f"{obj}_{idx}_{phase_label}.obj"
                    path = base / filename
                    if not path.exists():
                        raise FileNotFoundError(f"Missing: {path}")

                    arr = self._read_obj_file(path)
                    key_obj = "contours" if obj == "mesh" else obj

                    # Determine rest vs stress by idx:
                    state = "rest" if idx == phases["dia"] else "stress"
                    # e.g. "dia_contours_rest", "sys_wall_stress", etc.
                    attr_name = f"{phase_key}_{key_obj}_{state}"
                    setattr(self.patient_data, attr_name, arr)

    @staticmethod
    def _read_obj_file(obj_filename: Path) -> np.ndarray:
        obj_points: List[List[float]] = []
        with open(obj_filename, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    try:
                        x, y, z = map(float, parts[1:4])
                        obj_points.append([x, y, z])
                    except (ValueError, IndexError):
                        continue

        if not obj_points:
            raise ValueError(f"No vertices found in {obj_filename}")

        # Group points by z-value and sort contours from lowest to highest z
        z_groups: dict[float, list[list[float]]] = {}
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
