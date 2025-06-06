from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger


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
        self.pressure_rest: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        )  # Currently ignoring missing data by Optional should be handled later
        self.time_rest: Tuple[Optional[float], Optional[float]] = (None, None)
        self.pressure_stress: Tuple[Optional[float], Optional[float]] = (None, None)
        self.time_stress: Tuple[Optional[float], Optional[float]] = (None, None)
        self.loess_rest_dia: pd.DataFrame = pd.DataFrame()
        self.loess_rest_sys: pd.DataFrame = pd.DataFrame()
        self.loess_rest_global: pd.DataFrame = pd.DataFrame()
        self.loess_stress_dia: pd.DataFrame = pd.DataFrame()
        self.loess_stress_sys: pd.DataFrame = pd.DataFrame()
        self.loess_stress_global: pd.DataFrame = pd.DataFrame()

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
        self.pressure_dir, self.ivus_dir, self.loess_dir = self.find_dirs()
        self.patient_data = PatientData()

    def process_patient_data(self) -> PatientData:
        self.patient_data.id = self.id
        self.process_pressure()
        self.load_obj_data()
        self.load_loess_data()
        logger.info(f"Loaded PatientData for {self.id}")
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
            filename = f"{self.id}_pressure_{phase}_average_curve_all.csv"
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

    def load_loess_data(self) -> None:
        """Loads the Loess data for the patient."""
        loess_files = {
            "rest_local": "loess_data_rest.csv",
            "rest_global": "loess_data_rest_glob.csv",
            "stress_local": "loess_data_stress.csv",
            "stress_global": "loess_data_stress_glob.csv",
        }

        for key, filename in loess_files.items():
            filepath = self.loess_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing Loess file: {filepath}")
            if "rest_local" in key:
                df_dia = self._read_loess_phase(filepath, phase="D")
                df_sys = self._read_loess_phase(filepath, phase="S")
                self.patient_data.loess_rest_dia = df_dia
                self.patient_data.loess_rest_sys = df_sys
            elif "rest_global" in key:
                df = self._read_loess_global(filepath)
                self.patient_data.loess_rest_global = df
            elif "stress_local" in key:
                df_dia = self._read_loess_phase(filepath, phase="D")
                df_sys = self._read_loess_phase(filepath, phase="S")
                self.patient_data.loess_stress_dia = df_dia
                self.patient_data.loess_stress_sys = df_sys
            else:
                df = self._read_loess_global(filepath)
                self.patient_data.loess_stress_global = df

    @staticmethod
    def _read_loess_phase(filename: Path, phase: str = "D") -> pd.DataFrame:
        """
        Processes local Loess data for a specific phase (D or S), similar to _read_loess_global.
        """
        df = pd.read_csv(filename)
        df = df.dropna(axis=1, how="all").reset_index(drop=True)
        df = df[df["phase"] == phase] if "phase" in df.columns else df
        df["phase"] = phase  # Ensure phase column exists and is set

        measurement_blocks = {
            "lumen_area": ["lumen_area", "lumen_area_ci_lower", "lumen_area_ci_upper"],
            "elliptic_ratio": [
                "elliptic_ratio",
                "elliptic_ratio_ci_lower",
                "elliptic_ratio_ci_upper",
            ],
            "shortest_distance": [
                "shortest_distance",
                "shortest_ci_lower",
                "shortest_ci_upper",
            ],
        }

        dfs = []
        for measure, cols in measurement_blocks.items():
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols or "position" not in df.columns:
                continue
            block_df = df[["position", "phase"] + valid_cols].copy()
            block_df["measurement"] = measure
            column_map = {}
            if len(valid_cols) > 0:
                column_map[cols[0]] = "value"
            if len(valid_cols) > 1:
                column_map[cols[1]] = "ci_lower"
            if len(valid_cols) > 2:
                column_map[cols[2]] = "ci_upper"
            block_df = block_df.rename(columns=column_map)
            dfs.append(block_df.dropna(subset=["value"]))

        if not dfs:
            return pd.DataFrame(
                columns=["measurement", "phase", "value", "ci_lower", "ci_upper"]
            )
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["phase"] = (
            combined_df["phase"]
            .map({"D": "dia", "S": "sys"})
            .fillna(combined_df["phase"])
        )
        # Group by position and measurement, and compute mean for numeric columns
        grouped_df = (
            combined_df.groupby(["position", "measurement", "phase"], as_index=False)
            .agg({"value": "mean", "ci_lower": "mean", "ci_upper": "mean"})
            .set_index("position")
            .sort_index()
        )

        return grouped_df[["measurement", "phase", "value", "ci_lower", "ci_upper"]]

    @staticmethod
    def _read_loess_global(filename: Path) -> pd.DataFrame:
        """Processes global Loess data with multiple measurement types in one file."""
        # Read raw data and clean empty columns
        df = pd.read_csv(filename)
        df = df.dropna(axis=1, how="all").reset_index(drop=True)

        # Identify column groups for each measurement type
        measurement_blocks = {
            "lumen_area": ["lumen_area", "lumen_area_ci_lower", "lumen_area_ci_upper"],
            "elliptic_ratio": [
                "elliptic_ratio",
                "elliptic_ratio_ci_lower",
                "elliptic_ratio_ci_upper",
            ],
            "shortest_distance": [
                "shortest_distance",
                "shortest_ci_lower",
                "shortest_ci_upper",
            ],
        }

        # Process each measurement block
        dfs = []
        for measure, cols in measurement_blocks.items():
            # Find actual columns present in dataframe
            valid_cols = [c for c in cols if c in df.columns]

            block_df = df[["position", "phase"] + valid_cols].copy()
            block_df["measurement"] = measure

            # Rename columns to standardized format
            column_map = {cols[0]: "value", cols[1]: "ci_lower", cols[2]: "ci_upper"}
            block_df = block_df.rename(columns=column_map)

            dfs.append(block_df.dropna(subset=["value"]))

        # Combine all measurements and clean up
        combined_df = pd.concat(dfs, ignore_index=True)

        # Convert phase codes to labels
        combined_df["phase"] = combined_df["phase"].map({"D": "dia", "S": "sys"})

        # Set position as index and sort
        return (
            combined_df.set_index("position")
            .sort_index()
            .pipe(
                lambda df: df[["measurement", "phase", "value", "ci_lower", "ci_upper"]]
            )
        )
