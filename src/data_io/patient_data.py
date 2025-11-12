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

        Returns a tuple of (pressure_dir|None, ivus_dir|None, loess_dir|None).
        Missing directories are logged as warnings instead of raising.
        """
        processed_base_dir = self.working_dir / "processed"
        ivus_base_dir = self.working_dir / "3d_ivus"
        loess_base_dir = self.working_dir / "ivus"

        def find_matching_dir(base_dir: Path) -> Optional[Path]:
            if not base_dir.is_dir():
                logger.warning(f"Based directory does not exist: {base_dir}")
                return None
            for d in base_dir.iterdir():
                if d.is_dir() and self.id and self.id.lower() in d.name.lower():
                    return d
            logger.warning(f"No matching directory for id '{self.id}' in {base_dir}")
            return None

        pressure_dir: Optional[Path] = find_matching_dir(processed_base_dir)
        ivus_dir: Optional[Path] = find_matching_dir(ivus_base_dir)
        loess_dir: Optional[Path] = find_matching_dir(loess_base_dir)

        if pressure_dir is not None:
            m = re.search(r"(narco_[0-9]+)", pressure_dir.name, flags=re.IGNORECASE)
            if m:
                self.file_id = m.group(1).lower()  # e.g. "narco_303"
        else:
            logger.warning(
                f"No processed data directory for {self.id} in {processed_base_dir}; "
                "continuing with file_id set to provided id."
            )

        if ivus_dir is None:
            logger.warning(
                f"No IVUS data directory for {self.id} in {ivus_base_dir}; IVUS loading will be skipped."
            )
        if loess_dir is None:
            logger.warning(
                f"No Loess data directory for {self.id} in {loess_base_dir}; Loess-related data will be missing."
            )

        return pressure_dir, ivus_dir, loess_dir


    def process_pressure(self) -> None:
        patterns = ["rest_1", "dobu"]
        tuples_pressure: List[Tuple[Optional[float], Optional[float]]] = []
        tuples_time: List[Tuple[Optional[float], Optional[float]]] = []

        if self.pressure_dir is None or not self.pressure_dir.is_dir():
            logger.warning(
                f"Pressure directory not found for {self.id}; pressure values will be left as None."
            )
            # fill with (None, None) for both phases
            self.patient_data.pressure_rest = (None, None)
            self.patient_data.pressure_stress = (None, None)
            self.patient_data.time_rest = (None, None)
            self.patient_data.time_stress = (None, None)
            return

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

        # Ensure we always set both rest and stress even if some phases failed
        if len(tuples_pressure) >= 2:
            (self.patient_data.pressure_rest, self.patient_data.pressure_stress) = (
                tuples_pressure[0],
                tuples_pressure[1],
            )
            (self.patient_data.time_rest, self.patient_data.time_stress) = (
                tuples_time[0],
                tuples_time[1],
            )
        else:
            # fallback if something very odd happened
            self.patient_data.pressure_rest = tuples_pressure[0] if tuples_pressure else (None, None)
            self.patient_data.pressure_stress = tuples_pressure[1] if len(tuples_pressure) > 1 else (None, None)
            self.patient_data.time_rest = tuples_time[0] if tuples_time else (None, None)
            self.patient_data.time_stress = tuples_time[1] if len(tuples_time) > 1 else (None, None)

    def load_obj_data(self) -> None:
        """Loads the .obj files with the 3D aligned IVUS images and assigns to patient_data.
        Missing files/folders are warned about and skipped (attributes remain None).
        """
        if self.ivus_dir is None or not self.ivus_dir.is_dir():
            logger.warning(f"IVUS directory not available for {self.id}; skipping OBJ loading.")
            return

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
                logger.warning(f"No '{state}' directory in {self.ivus_dir}; skipping {state} files.")
                continue
            state_dir = candidates[0]

            for obj in objects:
                for phase, idx in phases.items():
                    filename = f"{obj}_{idx}_{state}.obj"
                    path = state_dir / filename
                    if not path.exists():
                        logger.warning(f"Missing: {path}; skipping this file.")
                        continue

                    try:
                        arr = self._read_obj_file(path)
                    except Exception as e:
                        logger.error(f"Failed to read {path}: {e}")
                        continue

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
                logger.warning(f"Missing comparison folder: {base}; skipping.")
                continue

            for obj in objects:
                for phase_suffix, idx in phases.items():  # phase_suffix is "dia"/"sys", idx "000"/"029"
                    filename = f"{obj}_{idx}_{phase_label}.obj"
                    path = base / filename
                    if not path.exists():
                        logger.warning(f"Missing: {path}; skipping.")
                        continue

                    try:
                        arr = self._read_obj_file(path)
                    except Exception as e:
                        logger.error(f"Failed to read {path}: {e}")
                        continue

                    key_obj = "contours" if obj == "mesh" else obj

                    # Determine rest vs stress by idx:
                    state = "rest" if idx == phases["dia"] else "stress"
                    # e.g. "dia_contours_rest", "sys_wall_stress", etc.
                    attr_name = f"{phase_key}_{key_obj}_{state}"
                    setattr(self.patient_data, attr_name, arr)

    @staticmethod
    def _read_obj_file(obj_filename: Path) -> np.ndarray:
        """
        Reads vertex lines from an .obj file and returns an ndarray with rows:
        [contour_idx, x, y, z]. If no vertices are found, returns an empty (0,4) array
        and logs a warning rather than raising.
        """
        obj_points: List[List[float]] = []
        try:
            with open(obj_filename, "r") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.split()
                        try:
                            x, y, z = map(float, parts[1:4])
                            obj_points.append([x, y, z])
                        except (ValueError, IndexError):
                            # skip malformed lines but continue parsing
                            continue
        except Exception as e:
            logger.error(f"Unable to open/read {obj_filename}: {e}")
            return np.empty((0, 4))

        if not obj_points:
            logger.warning(f"No vertices found in {obj_filename}; returning empty array.")
            return np.empty((0, 4))

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