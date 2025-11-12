from __future__ import annotations

import os
from typing import Dict
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_io.patient_data import PatientData
from loguru import logger
from preprocessing.contour_measurements import calculate_displacement_map
from preprocessing.contour_measurements import calculate_measurement_map
from preprocessing.contour_measurements import compute_contour_properties
from advanced_visualizations.plot_frame_diff import plot_frame_diff
from stats.plots import _plot_pca


class PatientPreprocessing:
    def __init__(self, patient_data: PatientData, output_dir: str, global_dir: str):
        self.patient_data = patient_data
        self.output_dir = os.path.join(output_dir, f"{patient_data.id}_stats")
        self.global_dir = global_dir

        # — normalize EVERY input‐z once, up front —
        self._normalize_all_contour_z()

        self._create_output_dir()
        self.df_rest = pd.DataFrame()
        self.df_stress = pd.DataFrame()
        self.df_dia = pd.DataFrame()
        self.df_sys = pd.DataFrame()
        self.patient_df: Dict = {}

    def _normalize_all_contour_z(self) -> None:
        """
        For each of the four arrays in patient_data:
          rest_contours_dia, rest_contours_sys,
          stress_contours_dia, stress_contours_sys
        subtract its own max-z and take abs, in place.
        This ensures that ostium is at position 0.
        """
        for attr in (
            "rest_contours_dia",
            "rest_contours_sys",
            "stress_contours_dia",
            "stress_contours_sys",
            "dia_contours_rest",
            "dia_contours_stress",
            "sys_contours_rest",
            "sys_contours_stress",
        ):
            arr = getattr(self.patient_data, attr, None)
            if isinstance(arr, np.ndarray) and arr.shape[1] > 3:
                z = arr[:, 3]
                arr[:, 3] = np.abs(z - z.max())
            else:
                logger.warning(
                    f"Cannot normalize z for `{attr}`: not found or wrong shape"
                )

    def _create_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")

    def process_case(self, plot=True) -> None:
        # 1) rest-vs-stress
        self.compute_lumen_changes()
        self.compute_sys_dia_properties(phase="rest")
        self.compute_sys_dia_properties(phase="stress")
        self.compute_sys_dia_properties(phase="dia_dia")
        self.compute_sys_dia_properties(phase="sys_sys")

        # 3) save patient data
        self.save_selected_to_global_stats()

        # 4) calculate distance maps.
        rest_path = os.path.join(self.output_dir, "rest_sys_dia_displacement_map.csv")
        stress_path = os.path.join(
            self.output_dir, "stress_sys_dia_displacement_map.csv"
        )
        dia_path = os.path.join(self.output_dir, "dia_dia_displacement_map.csv")
        sys_path = os.path.join(self.output_dir, "sys_sys_displacement_map.csv")

        calculate_displacement_map(
            self.patient_data.rest_contours_dia,
            self.patient_data.rest_contours_sys,
            self.patient_df["mean_intramural_length"][0],
            self.patient_df["pressure_change_pulsatile_rest"][0],
            self.patient_df["time_change_pulsatile_rest"][0],
            output_path=rest_path,
            adjust_pressure_time=False,
        )
        calculate_displacement_map(
            self.patient_data.stress_contours_dia,
            self.patient_data.stress_contours_sys,
            self.patient_df["mean_intramural_length"][0],
            self.patient_df["pressure_change_pulsatile_stress"][0],
            self.patient_df["time_change_pulsatile_stress"][0],
            output_path=stress_path,
            adjust_pressure_time=False,
        )
        calculate_displacement_map(
            self.patient_data.dia_contours_rest,
            self.patient_data.dia_contours_stress,
            self.patient_df["mean_intramural_length"][0],
            self.patient_df["pressure_change_stressind_dia"][0],
            output_path=dia_path,
            adjust_pressure_time=False,
        )
        calculate_displacement_map(
            self.patient_data.sys_contours_rest,
            self.patient_data.sys_contours_stress,
            self.patient_df["mean_intramural_length"][0],
            self.patient_df["pressure_change_stressind_sys"][0],
            output_path=sys_path,
            adjust_pressure_time=False,
        )
        calculate_measurement_map(
            df=self.df_rest,
            phase="rest",
            im_length=self.patient_df["mean_intramural_length"][0],
            pressure_change=self.patient_df["pressure_change_pulsatile_rest"][0],
            time_change=self.patient_df["time_change_pulsatile_rest"][0],
            output_path=self.output_dir,
            adjust_pressure_time=False,
        )
        calculate_measurement_map(
            df=self.df_stress,
            phase="stress",
            im_length=self.patient_df["mean_intramural_length"][0],
            pressure_change=self.patient_df["pressure_change_pulsatile_stress"][0],
            time_change=self.patient_df["time_change_pulsatile_stress"][0],
            output_path=self.output_dir,
            adjust_pressure_time=False,
        )
        calculate_measurement_map(
            df=self.df_dia,
            phase="dia_dia",
            im_length=self.patient_df["mean_intramural_length"][0],
            pressure_change=self.patient_df["pressure_change_stressind_dia"][0],
            time_change=None,
            output_path=self.output_dir,
            adjust_pressure_time=False,
        )
        calculate_measurement_map(
            df=self.df_sys,
            phase="sys_sys",
            im_length=self.patient_df["mean_intramural_length"][0],
            pressure_change=self.patient_df["pressure_change_stressind_sys"][0],
            time_change=None,
            output_path=self.output_dir,
            adjust_pressure_time=False,
        )
        if plot:
            plot_frame_diff(
                self.patient_data.rest_contours_dia,
                self.patient_data.dia_contours_rest,
                self.patient_data.rest_contours_sys,
                self.patient_data.sys_contours_rest,
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
                output_dir=self.output_dir,
            )

    def compute_lumen_changes(self) -> Dict[str, List[float]]:
        results = {}
        
        # Helper function to check if we have valid data
        def has_valid_data(data, min_length=2):
            return (data is not None and 
                    len(data) >= min_length and 
                    None not in data[:min_length])
        
        # rest
        if has_valid_data(self.patient_data.pressure_rest) and has_valid_data(self.patient_data.time_rest):
            dp = self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            dt = self.patient_data.time_rest[1] - self.patient_data.time_rest[0]
        else:
            dp = np.nan
            dt = np.nan
            
        if (self.patient_data.rest_contours_dia is not None and 
            self.patient_data.rest_contours_sys is not None):
            df = self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                self.patient_data.rest_contours_sys,
                dp,
                dt,
                mode="rest-stress",
                plot=False,
            )
            results["rest"] = df
        
        # stress
        if has_valid_data(self.patient_data.pressure_stress) and has_valid_data(self.patient_data.time_stress):
            dp = self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]
            dt = self.patient_data.time_stress[1] - self.patient_data.time_stress[0]
        else:
            dp = np.nan
            dt = np.nan
            
        if (self.patient_data.stress_contours_dia is not None and 
            self.patient_data.stress_contours_sys is not None):
            df = self._calculate_lumen_change(
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
                dp,
                dt,
                mode="rest-stress",
            )
            results["stress"] = df

        # dia-dia
        if (has_valid_data(self.patient_data.pressure_rest, 1) and 
            has_valid_data(self.patient_data.pressure_stress, 1)):
            dp = self.patient_data.pressure_stress[0] - self.patient_data.pressure_rest[0]
        else:
            dp = np.nan
            
        if (self.patient_data.dia_contours_rest is not None and 
            self.patient_data.dia_contours_stress is not None):
            df = self._calculate_lumen_change(
                self.patient_data.dia_contours_rest,
                self.patient_data.dia_contours_stress,
                dp,
                mode="dia-dia",
            )
            results["dia_dia"] = df

        # sys-sys
        if (has_valid_data(self.patient_data.pressure_rest, 2) and 
            has_valid_data(self.patient_data.pressure_stress, 2)):
            dp = self.patient_data.pressure_stress[1] - self.patient_data.pressure_rest[1]
        else:
            dp = np.nan
            
        if (self.patient_data.sys_contours_rest is not None and 
            self.patient_data.sys_contours_stress is not None):
            df = self._calculate_lumen_change(
                self.patient_data.sys_contours_rest,
                self.patient_data.sys_contours_stress,
                dp,
                mode="sys-sys",
            )
            results["sys_sys"] = df

        for cond, df in results.items():
            path = os.path.join(self.output_dir, f"{cond}_lumen_changes.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved {cond} lumen changes to {path}")
        return results

    @staticmethod
    def _pca_analysis(xy_spline, z, plot=False, title_str=""):
        """
        Principal component analysis of the 2D lumen contour

        -Parameters-
        xy_spline: 2D np.array of shape (n,2)
        z: Slice in [mm] being analyzed

        -Return-
        eigenvectors: as np.array of shape (2,2) with each eigenvector as column vector
        eigenvalues: as np.array of shape (2,)
        """

        centroid = np.mean(xy_spline, axis=0)
        centroid_3d = np.append(centroid, z)

        centered_data = xy_spline - centroid

        cov_matrix = np.cov(centered_data, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # only for debugging
        if plot:
            plt.figure(figsize=(6, 6))
            plt.plot(xy_spline[:, 0], xy_spline[:, 1], label=f"Contour z={z}")
            plt.scatter(*centroid, color="red", label="Centroid")
            plt.quiver(
                centroid[0],
                centroid[1],
                1,
                0,
                label=r"$e_x$",
                scale_units="xy",
                scale=1,
            )
            plt.quiver(
                centroid[0],
                centroid[1],
                0,
                1,
                label=r"$e_y$",
                scale_units="xy",
                scale=1,
            )

            scale = 1
            for i in range(2):
                vec = eigenvectors[:, i]
                plt.plot(
                    [centroid[0], centroid[0] + scale * vec[0]],
                    [centroid[1], centroid[1] + scale * vec[1]],
                    label=f"PC{i+1}",
                )

            plt.axis("equal")
            plt.legend(loc="upper right")
            plt.title(f"PCA of {title_str}")
            plt.show()

            print(f"{title_str}, PC1: {eigenvalues[0]}, PC2: {eigenvalues[1]}")

        return eigenvectors, eigenvalues, centroid_3d

    def _calculate_lumen_change(
        self,
        contours_dia: np.ndarray,
        contours_sys: np.ndarray,
        delta_pressure: float,
        delta_time: float = 1,
        mode: str = "rest-stress",
        plot: bool = False,
    ) -> List[float]:

        eigenvalues_dia_ls = []
        eigenvalues_sys_ls = []
        stretch_ls = []
        stretch_rate_ls = []
        stiffness_ls = []
        pc1_length_dia_ls = []
        pc1_length_sys_ls = []

        if contours_dia is None or contours_sys is None:
            logger.warning("Contours data is missing.")
            return stretch_ls

        # if length is different remove first rows of the longer one
        if contours_dia.shape[0] != contours_sys.shape[0]:
            min_len = min(contours_dia.shape[0], contours_sys.shape[0])
            if contours_dia.shape[0] > min_len:
                contours_dia = contours_dia[-min_len:]
            if contours_sys.shape[0] > min_len:
                contours_sys = contours_sys[-min_len:]

        dia_ids = np.unique(contours_dia[:, 0])
        sys_ids = np.unique(contours_sys[:, 0])

        if not np.array_equal(dia_ids, sys_ids):
            logger.error("Mismatch in contour IDs between diastole and systole")
            return stretch_ls

        for cid in dia_ids:
            d_pts = contours_dia[contours_dia[:, 0] == cid][:, 1:]
            s_pts = contours_sys[contours_sys[:, 0] == cid][:, 1:]
            if len(d_pts) != 501 or len(s_pts) != 501:
                logger.warning(f"Contour {cid} has incorrect point count")
                continue

            try:
                _, eigenvalues_dia, _ = self._pca_analysis(
                    d_pts[:, 0:2], int(cid), False, f"dia frame {int(cid)}"
                )
                _, eigenvalues_sys, _ = self._pca_analysis(
                    s_pts[:, 0:2], int(cid), False, f"sys frame {int(cid)}"
                )

                stretch = eigenvalues_sys[0] / eigenvalues_dia[0]
                stiffness = (eigenvalues_sys[0] / eigenvalues_dia[0]) / delta_pressure

                if mode == "rest-stress":
                    stretch_rate = (
                        eigenvalues_sys[0] / eigenvalues_dia[0]
                    ) / delta_time
                    stretch_rate_ls.append(stretch_rate)

                else:
                    stretch_rate_ls.append(np.nan)

                stretch_ls.append(stretch)
                stiffness_ls.append(stiffness)
                pc1_length_dia_ls.append(eigenvalues_dia[0])
                pc1_length_sys_ls.append(eigenvalues_sys[0])

                if plot:
                    eigenvalues_dia_ls.append(eigenvalues_dia)
                    eigenvalues_sys_ls.append(eigenvalues_sys)

            except Exception as e:
                logger.error(f"PCA failed for contour {cid}: {e}")
                stretch_ls.append(np.nan)
                stretch_rate_ls.append(np.nan)
                stiffness_ls.append(np.nan)
                pc1_length_dia_ls.append(np.nan)
                pc1_length_sys_ls.append(np.nan)

        df = pd.DataFrame(
            {
                "contour_id": range(len(stretch_ls)),
                "pc1_diastole": pc1_length_dia_ls,
                "pc1_systole": pc1_length_sys_ls,
                "stretch": stretch_ls,
                "stretch_rate": stretch_rate_ls,
                "stiffness": stiffness_ls,
            }
        )

        if plot:
            _plot_pca(eigenvalues_dia_ls, eigenvalues_sys_ls, dia_ids)

        return df

    def compute_sys_dia_properties(self, phase: str = "rest") -> None:
        """
        Compute systolic/diastolic properties, but if pressure/time are missing
        for the relevant phases, set the dp/dt-dependent columns (deltas) to 0.
        Results are saved to CSV (merged if existing) and assigned to
        self.df_rest / self.df_stress / self.df_dia / self.df_sys as appropriate.
        """
        # determine missing flags up front
        pr = self.patient_data.pressure_rest
        tr = self.patient_data.time_rest
        ps = self.patient_data.pressure_stress
        ts = self.patient_data.time_stress

        def _is_missing_pressure_time(p_tuple, t_tuple):
            # missing if tuple is None OR contains None OR non-finite values
            try:
                if p_tuple is None or t_tuple is None:
                    return True
                a, b = p_tuple
                ta, tb = t_tuple
                vals = [a, b, ta, tb]
                return any(v is None or not np.isfinite(float(v)) for v in vals)
            except Exception:
                return True

        missing_rest = _is_missing_pressure_time(pr, tr)
        missing_stress = _is_missing_pressure_time(ps, ts)

        # select arrays and compute dp/dt (or mark denom_invalid)
        if phase == "rest":
            dia = self.patient_data.rest_contours_dia
            sys = self.patient_data.rest_contours_sys
            dp = None if missing_rest else (float(pr[1]) - float(pr[0]))
            dt = None if missing_rest else (float(tr[1]) - float(tr[0]))
            denom_invalid = (dp is None or dt is None or not np.isfinite(dp) or not np.isfinite(dt) or dp * dt == 0.0)
            use_zero_deltas = denom_invalid
        elif phase == "stress":
            dia = self.patient_data.stress_contours_dia
            sys = self.patient_data.stress_contours_sys
            dp = None if missing_stress else (float(ps[1]) - float(ps[0]))
            dt = None if missing_stress else (float(ts[1]) - float(ts[0]))
            denom_invalid = (dp is None or dt is None or not np.isfinite(dp) or not np.isfinite(dt) or dp * dt == 0.0)
            use_zero_deltas = denom_invalid
        elif phase == "dia_dia":
            dia = self.patient_data.dia_contours_rest
            sys = self.patient_data.dia_contours_stress
            # dp uses rest[0] and stress[0]; if either rest or stress missing -> mark invalid
            if missing_rest or missing_stress:
                dp = None
                use_zero_deltas = True
            else:
                dp = float(ps[0]) - float(pr[0])
                use_zero_deltas = (not np.isfinite(dp) or dp == 0.0)
            dt = 1.0
        elif phase == "sys_sys":
            dia = self.patient_data.sys_contours_rest
            sys = self.patient_data.sys_contours_stress
            if missing_rest or missing_stress:
                dp = None
                use_zero_deltas = True
            else:
                dp = float(ps[1]) - float(pr[1])
                use_zero_deltas = (not np.isfinite(dp) or dp == 0.0)
            dt = 1.0
        else:
            logger.error(f"Invalid phase: {phase}, expected 'rest', 'stress', 'dia_dia' or 'sys_sys'")
            return

        # If denom invalid -> we'll set delta columns to 0
        if use_zero_deltas:
            logger.warning(f"Setting delta columns to 0 for {self.patient_data.id} ({phase}) due to missing/invalid pressure/time.")

        # compute contour properties safely
        def _safe_props(arr, name):
            try:
                if arr is None:
                    logger.warning(f"{name} is None for {self.patient_data.id} ({phase})")
                    return np.empty((0, 7))
                props = compute_contour_properties(arr)
                if props is None:
                    return np.empty((0, 7))
                props = np.asarray(props, dtype=float)
                if props.ndim != 2 or props.shape[1] < 7:
                    logger.warning(f"{name} has unexpected shape {props.shape} for {self.patient_data.id} ({phase}); expected Nx>=7.")
                    return np.empty((0, 7))
                return props
            except Exception as e:
                logger.warning(f"Failed to compute contour properties for {name} ({self.patient_data.id}, {phase}): {e}")
                return np.empty((0, 7))

        p_dia = _safe_props(dia, "dia")
        p_sys = _safe_props(sys, "sys")

        # expected extra columns per phase
        if p_dia.size == 0 or p_sys.size == 0:
            # make empty DataFrame with columns for this phase (so downstream code still works)
            common_cols = ["contour_id", "z_value", "delta_lumen_area", "delta_min_dist", "delta_max_dist", "delta_elliptic_ratio"]
            if phase in ("rest", "stress"):
                extra = [
                    "lumen_area_dia", "lumen_area_sys",
                    "min_dist_dia", "min_dist_sys",
                    "max_dist_dia", "max_dist_sys",
                    "elliptic_ratio_dia", "elliptic_ratio_sys",
                    "circ_dia", "circ_sys",
                ]
            elif phase == "dia_dia":
                extra = [
                    "lumen_area_dia_rest", "lumen_area_dia_stress",
                    "min_dist_dia_rest", "min_dist_dia_stress",
                    "max_dist_dia_rest", "max_dist_dia_stress",
                    "elliptic_ratio_dia_rest", "elliptic_ratio_dia_stress",
                    "circ_dia_rest", "circ_dia_stress",
                ]
            else:  # sys_sys
                extra = [
                    "lumen_area_sys_rest", "lumen_area_sys_stress",
                    "min_dist_sys_rest", "min_dist_sys_stress",
                    "max_dist_sys_rest", "max_dist_sys_stress",
                    "elliptic_ratio_sys_rest", "elliptic_ratio_sys_stress",
                    "circ_sys_rest", "circ_sys_stress",
                ]
            df_empty = pd.DataFrame(columns=common_cols + extra)
            # still attempt to merge with existing CSV if present, then store attribute
            in_path = os.path.join(self.output_dir, f"{phase}_lumen_changes.csv")
            if os.path.exists(in_path):
                lum = pd.read_csv(in_path)
                df_empty = df_empty.merge(lum, on="contour_id", how="left")
                df_empty.to_csv(in_path, index=False)
            if phase == "rest":
                self.df_rest = df_empty
            elif phase == "stress":
                self.df_stress = df_empty
            elif phase == "dia_dia":
                self.df_dia = df_empty
            elif phase == "sys_sys":
                self.df_sys = df_empty
            return

        # align rows (truncate to shortest)
        n_rows = min(p_dia.shape[0], p_sys.shape[0])
        if p_dia.shape[0] != p_sys.shape[0]:
            logger.warning(f"p_dia rows ({p_dia.shape[0]}) != p_sys rows ({p_sys.shape[0]}) for {self.patient_data.id} ({phase}); truncating to {n_rows} rows.")
        p_dia = p_dia[:n_rows, :]
        p_sys = p_sys[:n_rows, :]

        # compute deltas: either zero (when use_zero_deltas) or (p_sys - p_dia) / (dp*dt)
        if use_zero_deltas:
            delta_area = np.zeros(n_rows)
            delta_min = np.zeros(n_rows)
            delta_max = np.zeros(n_rows)
            delta_ell = np.zeros(n_rows)
        else:
            denom = float(dp) * float(dt)
            if not np.isfinite(denom) or denom == 0.0:
                logger.warning(f"Denominator dp*dt invalid for {self.patient_data.id} ({phase}); setting delta columns to 0.")
                delta_area = np.zeros(n_rows)
                delta_min = np.zeros(n_rows)
                delta_max = np.zeros(n_rows)
                delta_ell = np.zeros(n_rows)
            else:
                delta_area = (p_sys[:n_rows, 2] - p_dia[:n_rows, 2]) / denom
                delta_min = (p_sys[:n_rows, 3] - p_dia[:n_rows, 3]) / denom
                delta_max = (p_sys[:n_rows, 4] - p_dia[:n_rows, 4]) / denom
                delta_ell = (p_sys[:n_rows, 5] - p_dia[:n_rows, 5]) / denom

        # Build DataFrame according to phase (keeps your original column layout)
        if phase in ("rest", "stress"):
            df = pd.DataFrame(
                {
                    "contour_id": p_dia[:, 0].astype(int),
                    "z_value": p_dia[:, 1],
                    "delta_lumen_area": delta_area,
                    "delta_min_dist": delta_min,
                    "delta_max_dist": delta_max,
                    "delta_elliptic_ratio": delta_ell,
                    "lumen_area_dia": p_dia[:, 2],
                    "lumen_area_sys": p_sys[:, 2],
                    "min_dist_dia": p_dia[:, 3],
                    "min_dist_sys": p_sys[:, 3],
                    "max_dist_dia": p_dia[:, 4],
                    "max_dist_sys": p_sys[:, 4],
                    "elliptic_ratio_dia": p_dia[:, 5],
                    "elliptic_ratio_sys": p_sys[:, 5],
                    "circ_dia": p_dia[:, 6],
                    "circ_sys": p_sys[:, 6],
                }
            )
        elif phase == "dia_dia":
            df = pd.DataFrame(
                {
                    "contour_id": p_dia[:, 0].astype(int),
                    "z_value": p_dia[:, 1],
                    "delta_lumen_area": delta_area,
                    "delta_min_dist": delta_min,
                    "delta_max_dist": delta_max,
                    "delta_elliptic_ratio": delta_ell,
                    "lumen_area_dia_rest": p_dia[:, 2],
                    "lumen_area_dia_stress": p_sys[:, 2],
                    "min_dist_dia_rest": p_dia[:, 3],
                    "min_dist_dia_stress": p_sys[:, 3],
                    "max_dist_dia_rest": p_dia[:, 4],
                    "max_dist_dia_stress": p_sys[:, 4],
                    "elliptic_ratio_dia_rest": p_dia[:, 5],
                    "elliptic_ratio_dia_stress": p_sys[:, 5],
                    "circ_dia_rest": p_dia[:, 6],
                    "circ_dia_stress": p_sys[:, 6],
                }
            )
        else:  # sys_sys
            df = pd.DataFrame(
                {
                    "contour_id": p_dia[:, 0].astype(int),
                    "z_value": p_dia[:, 1],
                    "delta_lumen_area": delta_area,
                    "delta_min_dist": delta_min,
                    "delta_max_dist": delta_max,
                    "delta_elliptic_ratio": delta_ell,
                    "lumen_area_sys_rest": p_dia[:, 2],
                    "lumen_area_sys_stress": p_sys[:, 2],
                    "min_dist_sys_rest": p_dia[:, 3],
                    "min_dist_sys_stress": p_sys[:, 3],
                    "max_dist_sys_rest": p_dia[:, 4],
                    "max_dist_sys_stress": p_sys[:, 4],
                    "elliptic_ratio_sys_rest": p_dia[:, 5],
                    "elliptic_ratio_sys_stress": p_sys[:, 5],
                    "circ_sys_rest": p_dia[:, 6],
                    "circ_sys_stress": p_sys[:, 6],
                }
            )

        # CSV merge + save logic (keeps your behaviour)
        in_path = os.path.join(self.output_dir, f"{phase}_lumen_changes.csv")
        if os.path.exists(in_path):
            try:
                lum = pd.read_csv(in_path)
                df = df.merge(lum, on="contour_id", how="left")
                df.to_csv(in_path, index=False)
            except Exception as e:
                logger.error(f"Failed to merge/write {in_path}: {e}")
        else:
            logger.warning(f"Missing file: {in_path}")

        # store in object for later use
        if phase == "rest":
            self.df_rest = df
        elif phase == "stress":
            self.df_stress = df
        elif phase == "dia_dia":
            self.df_dia = df
        elif phase == "sys_sys":
            self.df_sys = df

    def save_selected_to_global_stats(self):
        """Extensive summary functions for patients individual stats into a global dataframe.
            Column definition are explicit, allows for easy updates.
            Input:
            - self.df_rest, self.df_stress, self.df_dia, self.df_sys
        - Output:
            - DataFrame with selected statistics, saved to global stats directory.
        """
        for df_name in ['df_rest', 'df_stress', 'df_dia', 'df_sys']:
            df = getattr(self, df_name)
            if not df.empty:
                missing_cols = ['stretch', 'stretch_rate', 'stiffness', 'pc1_diastole', 'pc1_systole']
                for col in missing_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                        logger.warning(f"Added missing column '{col}' to {df_name} for patient {self.patient_data.id}")

        # Helper to find first hit >= threshold, returns positional index or None
        def first_hit(df, col, phase_desc):
            hits = df.index[df[col] >= 1.3]
            if len(hits) == 0:
                logger.warning(
                    f"{self.patient_data.id}: no values >=1.3 for '{col}' in {phase_desc}; skipping stats"
                )
                return None
            return hits[0]

        # 1) Determine ostium labels and their positional indices
        ost_id_rest = self.df_rest["contour_id"].max()
        ost_id_stress = self.df_stress["contour_id"].max()
        ost_id_dia = self.df_dia["contour_id"].max()
        ost_id_sys = self.df_sys["contour_id"].max()

        try:
            ost_pos_rest = self.df_rest.index[
                self.df_rest["contour_id"] == ost_id_rest
            ][0]
            ost_pos_stress = self.df_stress.index[
                self.df_stress["contour_id"] == ost_id_stress
            ][0]
            ost_pos_dia = self.df_dia.index[self.df_dia["contour_id"] == ost_id_dia][0]
            ost_pos_sys = self.df_sys.index[self.df_sys["contour_id"] == ost_id_sys][0]
        except IndexError as e:
            logger.error(
                f"{self.patient_data.id}: could not locate ostium row by label: {e}"
            )
            return

        # 2) Find first-hit indices for elliptic_ratio >= 1.3 in each phase
        im_pos_rest = None

        d_rest = first_hit(self.df_rest, "elliptic_ratio_dia", "rest-dia")
        s_rest = first_hit(self.df_rest, "elliptic_ratio_sys", "rest-sys")
        if d_rest is not None and s_rest is not None:
            im_pos_rest = (d_rest + s_rest) // 2
        elif d_rest is not None and s_rest is None:
            im_pos_rest = d_rest
        elif d_rest is None and s_rest is not None:
            im_pos_rest = s_rest
        else:
            return

        im_pos_stress = None
        d_str = first_hit(self.df_stress, "elliptic_ratio_dia", "stress-dia")
        s_str = first_hit(self.df_stress, "elliptic_ratio_sys", "stress-sys")
        if d_str is not None and s_str is not None:
            im_pos_stress = (d_str + s_str) // 2
        elif d_str is not None and s_str is None:
            im_pos_stress = d_str
        elif d_str is None and s_str is not None:
            im_pos_stress = s_str
        else:
            return

        im_pos_dia = None
        d_dia = first_hit(self.df_dia, "elliptic_ratio_dia_rest", "dia-rest")
        s_dia = first_hit(self.df_dia, "elliptic_ratio_dia_stress", "dia-stress")
        if d_dia is not None and s_dia is not None:
            im_pos_dia = (d_dia + s_dia) // 2
        elif d_dia is not None and s_dia is None:
            im_pos_dia = d_dia
        elif d_dia is None and s_dia is not None:
            im_pos_dia = s_dia
        else:
            return

        im_pos_sys = None
        d_sys = first_hit(self.df_sys, "elliptic_ratio_sys_rest", "sys-rest")
        s_sys = first_hit(self.df_sys, "elliptic_ratio_sys_stress", "sys-stress")
        if d_sys is not None and s_sys is not None:
            im_pos_sys = (d_sys + s_sys) // 2
        elif d_sys is not None and s_sys is None:
            im_pos_sys = d_sys
        elif d_sys is None and s_sys is not None:
            im_pos_sys = s_sys
        else:
            return

        # 3) Read z_value at these positions
        im_len_rest = self.df_rest.iloc[im_pos_rest]["z_value"]
        im_len_stress = self.df_stress.iloc[im_pos_stress]["z_value"]
        im_len_dia = self.df_dia.iloc[im_pos_dia]["z_value"]
        im_len_sys = self.df_sys.iloc[im_pos_sys]["z_value"]

        # 4) Compute mean
        # im_len = (im_len_rest + im_len_stress + im_len_dia + im_len_sys) / 4
        im_len = (im_len_rest + im_len_stress) / 2

        # needed for sectionwise
        im_20_per = im_len * 0.2
        n_intramural = round(im_len / im_20_per)
        n_segments = (
            n_intramural + 3
        )  # number of segments for extramural, 3 based on data

        section_stats = {}

        for phase, df in (
            ("pulsatile_rest", self.df_rest),
            ("pulsatile_stress", self.df_stress),
            ("stressind_dia", self.df_dia),
            ("stressind_sys", self.df_sys),
        ):
            for i in range(n_segments):
                z0 = i * im_20_per
                z1 = (i + 1) * im_20_per
                sel = df[(df["z_value"] >= z0) & (df["z_value"] < z1)]

                if sel.empty:
                    mean_values = {k: np.nan for k in [
                        "lumen", "min_dist", "ellip_ratio", "stretch", "stretch_rate",
                        "stiffness", "pc1_dia", "pc1_sys",
                        "lumen_dia", "lumen_sys", "min_dist_dia", "min_dist_sys"
                    ]}
                else:
                    # base metrics (always the same column names)
                    mean_values = {
                        "lumen": sel["delta_lumen_area"].mean(),
                        "min_dist": sel["delta_min_dist"].mean(),
                        "ellip_ratio": sel["delta_elliptic_ratio"].mean(),
                        "stretch": sel["stretch"].mean(),
                        "stretch_rate": sel["stretch_rate"].mean(),
                        "stiffness": sel["stiffness"].mean(),
                        "pc1_dia": sel["pc1_diastole"].mean(),
                        "pc1_sys": sel["pc1_systole"].mean(),
                    }

                    # candidate column mappings for dia/sys values
                    mapping = {
                        "lumen_dia": ["lumen_area_dia", "lumen_area_sys_rest", "lumen_area_dia_rest"],
                        "lumen_sys": ["lumen_area_sys", "lumen_area_sys_stress", "lumen_area_dia_stress"],
                        "min_dist_dia": ["min_dist_dia", "min_dist_sys_rest", "min_dist_dia_rest"],
                        "min_dist_sys": ["min_dist_sys", "min_dist_sys_stress", "min_dist_dia_stress"],
                    }

                    # loop through mapping: take first matching column
                    for key, candidates in mapping.items():
                        for c in candidates:
                            if c in sel.columns:
                                mean_values[key] = sel[c].mean()
                                break
                        else:
                            mean_values[key] = np.nan  # no match

                pct_label = (i + 1) * 20
                for key, value in mean_values.items():
                    section_stats[f"{phase}_pct_{pct_label}_{key}"] = value

        # 5) Compute MLA positions, clamped
        def clamp(idx, df):
            return max(0, min(idx, len(df) - 1))

        # mla_pos_rest = clamp((ost_pos_rest + im_pos_rest) // 2, self.df_rest)
        # mla_pos_stress = clamp((ost_pos_stress + im_pos_stress) // 2, self.df_stress)
        # mla_pos_dia = clamp((ost_pos_dia + im_pos_dia) // 2, self.df_dia)
        # mla_pos_sys = clamp((ost_pos_sys + im_pos_sys) // 2, self.df_sys)

        mla_pos_rest = self.df_rest["lumen_area_sys"].idxmin()
        mla_pos_stress = self.df_stress["lumen_area_sys"].idxmin()
        mla_pos_dia = self.df_dia["lumen_area_dia_stress"].idxmin()
        mla_pos_sys = self.df_sys["lumen_area_sys_stress"].idxmin()

        # reference position for rest and stress the location, where the greates lumen area is found
        def _ref_pos_from_df(df, candidates):
            available = [c for c in candidates if c in df.columns]
            if not available:
                logger.warning(f"{self.patient_data.id}: no lumen area colfumns found in dataframe")
                return 0
            
            if len(available) == 1:
                return df[available[0]].idxmax()
            # row-wise max between the available lumen columns, then global max index
            return df[available].max(axis=1).idxmax()

        ref_pos_rest = _ref_pos_from_df(self.df_rest, ["lumen_area_dia", "lumen_area_sys"])
        ref_pos_stress = _ref_pos_from_df(self.df_stress, ["lumen_area_dia", "lumen_area_sys"])

        # 6) Build the single-row DataFrame
        data = {
            "patient_id": [self.patient_data.id],
            "mean_intramural_length": [im_len],
            "pressure_change_pulsatile_rest": [
                (self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0] 
                if (self.patient_data.pressure_rest and 
                    len(self.patient_data.pressure_rest) >= 2 and
                    self.patient_data.pressure_rest[0] is not None and
                    self.patient_data.pressure_rest[1] is not None)
                else None)
            ],
            "time_change_pulsatile_rest": [
                (self.patient_data.time_rest[1] - self.patient_data.time_rest[0]
                if (self.patient_data.time_rest and 
                    len(self.patient_data.time_rest) >= 2 and
                    self.patient_data.time_rest[0] is not None and
                    self.patient_data.time_rest[1] is not None)
                else None)
            ],
            "pressure_change_pulsatile_stress": [
                (self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]
                if (self.patient_data.pressure_stress and 
                    len(self.patient_data.pressure_stress) >= 2 and
                    self.patient_data.pressure_stress[0] is not None and
                    self.patient_data.pressure_stress[1] is not None)
                else None)
            ],
            "time_change_pulsatile_stress": [
                (self.patient_data.time_stress[1] - self.patient_data.time_stress[0]
                if (self.patient_data.time_stress and 
                    len(self.patient_data.time_stress) >= 2 and
                    self.patient_data.time_stress[0] is not None and
                    self.patient_data.time_stress[1] is not None)
                else None)
            ],
            "pressure_change_stressind_dia": [
                (self.patient_data.pressure_stress[0] - self.patient_data.pressure_rest[0]
                if (self.patient_data.pressure_rest and 
                    self.patient_data.pressure_stress and
                    len(self.patient_data.pressure_rest) >= 1 and
                    len(self.patient_data.pressure_stress) >= 1 and
                    self.patient_data.pressure_rest[0] is not None and
                    self.patient_data.pressure_stress[0] is not None)
                else None)
            ],
            "pressure_change_stressind_sys": [
                (self.patient_data.pressure_stress[1] - self.patient_data.pressure_rest[1]
                if (self.patient_data.pressure_rest and 
                    self.patient_data.pressure_stress and
                    len(self.patient_data.pressure_rest) >= 2 and
                    len(self.patient_data.pressure_stress) >= 2 and
                    self.patient_data.pressure_rest[1] is not None and
                    self.patient_data.pressure_stress[1] is not None)
                else None)
            ],
            "pulsatile_rest_lumen_ost": self.df_rest.iloc[ost_pos_rest][
                "delta_lumen_area"
            ],
            "pulsatile_rest_min_ost": self.df_rest.iloc[ost_pos_rest]["delta_min_dist"],
            "pulsatile_rest_ellip_ost": self.df_rest.iloc[ost_pos_rest][
                "delta_elliptic_ratio"
            ],
            "pulsatile_rest_lumen_percent_ost": (self.df_rest.iloc[ost_pos_rest]["lumen_area_dia"] - self.df_rest.iloc[ost_pos_rest][
                "lumen_area_sys"
            ])
            / self.df_rest.iloc[ost_pos_rest]["lumen_area_dia"],
            "pulsatile_rest_min_percent_ost": (self.df_rest.iloc[ost_pos_rest]["min_dist_dia"] - self.df_rest.iloc[ost_pos_rest][
                "min_dist_sys"
            ])
            / self.df_rest.iloc[ost_pos_rest]["min_dist_dia"],
            "pulsatile_rest_stretch_ost": self.df_rest.iloc[ost_pos_rest]["stretch"],
            "pulsatile_rest_stretch_rate_ost": self.df_rest.iloc[ost_pos_rest][
                "stretch_rate"
            ],
            "pulsatile_rest_stiffness_ost": self.df_rest.iloc[ost_pos_rest][
                "stiffness"
            ],
            "pulsatile_rest_pc1_diastole_ost": self.df_rest.iloc[ost_pos_rest][
                "pc1_diastole"
            ],
            "pulsatile_rest_pc1_systole_ost": self.df_rest.iloc[ost_pos_rest][
                "pc1_systole"
            ],
            "pulsatile_rest_lumen_mla": self.df_rest.iloc[mla_pos_rest][
                "delta_lumen_area"
            ],
            "pulsatile_rest_min_mla": self.df_rest.iloc[mla_pos_rest]["delta_min_dist"],
            "pulsatile_rest_ellip_mla": self.df_rest.iloc[mla_pos_rest][
                "delta_elliptic_ratio"
            ],
            "pulsatile_rest_lumen_percent_mla": (self.df_rest.iloc[mla_pos_rest]["lumen_area_dia"] - self.df_rest.iloc[mla_pos_rest][
                "lumen_area_sys"
            ])
            / self.df_rest.iloc[mla_pos_rest]["lumen_area_dia"],
            "pulsatile_rest_min_percent_mla": (self.df_rest.iloc[mla_pos_rest]["min_dist_dia"] - self.df_rest.iloc[mla_pos_rest][
                "min_dist_sys"
            ])
            / self.df_rest.iloc[mla_pos_rest]["min_dist_dia"],
            "pulsatile_rest_stretch_mla": self.df_rest.iloc[mla_pos_rest]["stretch"],
            "pulsatile_rest_stretch_rate_mla": self.df_rest.iloc[mla_pos_rest][
                "stretch_rate"
            ],
            "pulsatile_rest_stiffness_mla": self.df_rest.iloc[mla_pos_rest][
                "stiffness"
            ],
            "pulsatile_rest_pc1_diastole_mla": self.df_rest.iloc[mla_pos_rest][
                "pc1_diastole"
            ],
            "pulsatile_rest_pc1_systole_mla": self.df_rest.iloc[mla_pos_rest][
                "pc1_systole"
            ],
            "pulsatile_stress_lumen_ost": self.df_stress.iloc[ost_pos_stress][
                "delta_lumen_area"
            ],
            "pulsatile_stress_min_ost": self.df_stress.iloc[ost_pos_stress][
                "delta_min_dist"
            ],
            "pulsatile_stress_ellip_ost": self.df_stress.iloc[ost_pos_stress][
                "delta_elliptic_ratio"
            ],
            "pulsatile_stress_lumen_percent_ost": (self.df_stress.iloc[ost_pos_stress]["lumen_area_dia"] - self.df_stress.iloc[ost_pos_stress][
                "lumen_area_sys"
            ])
            / self.df_stress.iloc[ost_pos_stress]["lumen_area_dia"],
            "pulsatile_stress_min_percent_ost": (self.df_stress.iloc[ost_pos_stress]["min_dist_dia"] - self.df_stress.iloc[ost_pos_stress][
                "min_dist_sys"
            ])
            / self.df_stress.iloc[ost_pos_stress]["min_dist_dia"],
            "pulsatile_stress_stretch_ost": self.df_stress.iloc[ost_pos_stress][
                "stretch"
            ],
            "pulsatile_stress_stretch_rate_ost": self.df_stress.iloc[ost_pos_stress][
                "stretch_rate"
            ],
            "pulsatile_stress_stiffness_ost": self.df_stress.iloc[ost_pos_stress][
                "stiffness"
            ],
            "pulsatile_stress_pc1_diastole_ost": self.df_stress.iloc[ost_pos_stress][
                "pc1_diastole"
            ],
            "pulsatile_stress_pc1_systole_ost": self.df_stress.iloc[ost_pos_stress][
                "pc1_systole"
            ],
            "pulsatile_stress_lumen_mla": self.df_stress.iloc[mla_pos_stress][
                "delta_lumen_area"
            ],
            "pulsatile_stress_min_mla": self.df_stress.iloc[mla_pos_stress][
                "delta_min_dist"
            ],
            "pulsatile_stress_ellip_mla": self.df_stress.iloc[mla_pos_stress][
                "delta_elliptic_ratio"
            ],
            "pulsatile_stress_lumen_percent_mla": (self.df_stress.iloc[mla_pos_stress]["lumen_area_dia"] - self.df_stress.iloc[mla_pos_stress][
                "lumen_area_sys"
            ])
            / self.df_stress.iloc[mla_pos_stress]["lumen_area_dia"],
            "pulsatile_stress_min_percent_mla": (self.df_stress.iloc[mla_pos_stress]["min_dist_dia"] - self.df_stress.iloc[mla_pos_stress][
                "min_dist_sys"
            ])
            / self.df_stress.iloc[mla_pos_stress]["min_dist_dia"],
            "pulsatile_stress_stretch_mla": self.df_stress.iloc[mla_pos_stress][
                "stretch"
            ],
            "pulsatile_stress_stretch_rate_mla": self.df_stress.iloc[mla_pos_stress][
                "stretch_rate"
            ],
            "pulsatile_stress_stiffness_mla": self.df_stress.iloc[mla_pos_stress][
                "stiffness"
            ],
            "pulsatile_stress_pc1_diastole_mla": self.df_stress.iloc[mla_pos_stress][
                "pc1_diastole"
            ],
            "pulsatile_stress_pc1_systole_mla": self.df_stress.iloc[mla_pos_stress][
                "pc1_systole"
            ],
            "stressind_dia_lumen_ost": self.df_dia.iloc[ost_pos_dia][
                "delta_lumen_area"
            ],
            "stressind_dia_min_ost": self.df_dia.iloc[ost_pos_dia]["delta_min_dist"],
            "stressind_dia_ellip_ost": self.df_dia.iloc[ost_pos_dia][
                "delta_elliptic_ratio"
            ],
            "stressind_dia_lumen_percent_ost": (self.df_dia.iloc[ost_pos_dia]["lumen_area_dia_rest"] - self.df_dia.iloc[ost_pos_dia][
                "lumen_area_dia_stress"
            ])
            / self.df_dia.iloc[ost_pos_dia]["lumen_area_dia_rest"],
            "stressind_dia_min_percent_ost": (self.df_dia.iloc[ost_pos_dia]["min_dist_dia_rest"] - self.df_dia.iloc[ost_pos_dia][
                "min_dist_dia_stress"
            ])
            / self.df_dia.iloc[ost_pos_dia]["min_dist_dia_rest"],
            "stressind_dia_stretch_ost": self.df_dia.iloc[ost_pos_dia]["stretch"],
            "stressind_dia_stretch_rate_ost": self.df_dia.iloc[ost_pos_dia][
                "stretch_rate"
            ],
            "stressind_dia_stiffness_ost": self.df_dia.iloc[ost_pos_dia]["stiffness"],
            "stressind_dia_pc1_diastole_ost": self.df_dia.iloc[ost_pos_dia][
                "pc1_diastole"
            ],
            "stressind_dia_pc1_systole_ost": self.df_dia.iloc[ost_pos_dia][
                "pc1_systole"
            ],
            "stressind_dia_lumen_mla": self.df_dia.iloc[mla_pos_dia][
                "delta_lumen_area"
            ],
            "stressind_dia_min_mla": self.df_dia.iloc[mla_pos_dia]["delta_min_dist"],
            "stressind_dia_ellip_mla": self.df_dia.iloc[mla_pos_dia][
                "delta_elliptic_ratio"
            ],
            "stressind_dia_lumen_percent_mla": (self.df_dia.iloc[mla_pos_dia]["lumen_area_dia_rest"] - self.df_dia.iloc[mla_pos_dia][
                "lumen_area_dia_stress"
            ])
            / self.df_dia.iloc[mla_pos_dia]["lumen_area_dia_rest"],
            "stressind_dia_min_percent_mla": (self.df_dia.iloc[mla_pos_dia]["min_dist_dia_rest"] - self.df_dia.iloc[mla_pos_dia][
                "min_dist_dia_stress"
            ])
            / self.df_dia.iloc[mla_pos_dia]["min_dist_dia_rest"],
            "stressind_dia_stretch_mla": self.df_dia.iloc[mla_pos_dia]["stretch"],
            "stressind_dia_stretch_rate_mla": self.df_dia.iloc[mla_pos_dia][
                "stretch_rate"
            ],
            "stressind_dia_stiffness_mla": self.df_dia.iloc[mla_pos_dia]["stiffness"],
            "stressind_dia_pc1_diastole_mla": self.df_dia.iloc[mla_pos_dia][
                "pc1_diastole"
            ],
            "stressind_dia_pc1_systole_mla": self.df_dia.iloc[mla_pos_dia][
                "pc1_systole"
            ],
            "stressind_sys_lumen_ost": self.df_sys.iloc[ost_pos_sys][
                "delta_lumen_area"
            ],
            "stressind_sys_min_ost": self.df_sys.iloc[ost_pos_sys]["delta_min_dist"],
            "stressind_sys_ellip_ost": self.df_sys.iloc[ost_pos_sys][
                "delta_elliptic_ratio"
            ],
            "stressind_sys_lumen_percent_ost": (self.df_sys.iloc[ost_pos_sys]["lumen_area_sys_rest"] - self.df_sys.iloc[ost_pos_sys][
                "lumen_area_sys_stress"
            ])
            / self.df_sys.iloc[ost_pos_sys]["lumen_area_sys_rest"],
            "stressind_sys_min_percent_ost": (self.df_sys.iloc[ost_pos_sys]["max_dist_sys_rest"] - self.df_sys.iloc[ost_pos_sys][
                "min_dist_sys_stress"
            ])
            / self.df_sys.iloc[ost_pos_sys]["max_dist_sys_rest"],
            "stressind_sys_stretch_ost": self.df_sys.iloc[ost_pos_sys]["stretch"],
            "stressind_sys_stretch_rate_ost": self.df_sys.iloc[ost_pos_sys][
                "stretch_rate"
            ],
            "stressind_sys_stiffness_ost": self.df_sys.iloc[ost_pos_sys]["stiffness"],
            "stressind_sys_pc1_diastole_ost": self.df_sys.iloc[ost_pos_sys][
                "pc1_diastole"
            ],
            "stressind_sys_pc1_systole_ost": self.df_sys.iloc[ost_pos_sys][
                "pc1_systole"
            ],
            "stressind_sys_lumen_mla": self.df_sys.iloc[mla_pos_sys][
                "delta_lumen_area"
            ],
            "stressind_sys_min_mla": self.df_sys.iloc[mla_pos_sys]["delta_min_dist"],
            "stressind_sys_ellip_mla": self.df_sys.iloc[mla_pos_sys][
                "delta_elliptic_ratio"
            ],
            "stressind_sys_lumen_percent_mla": (self.df_sys.iloc[mla_pos_sys]["lumen_area_sys_rest"] - self.df_sys.iloc[mla_pos_sys][
                "lumen_area_sys_stress"
            ])
            / self.df_sys.iloc[mla_pos_sys]["lumen_area_sys_rest"],
            "stressind_sys_min_percent_mla": (self.df_sys.iloc[mla_pos_sys]["max_dist_sys_rest"] - self.df_sys.iloc[mla_pos_sys][
                "min_dist_sys_stress"
            ])
            / self.df_sys.iloc[mla_pos_sys]["max_dist_sys_rest"],
            "stressind_sys_stretch_mla": self.df_sys.iloc[mla_pos_sys]["stretch"],
            "stressind_sys_stretch_rate_mla": self.df_sys.iloc[mla_pos_sys][
                "stretch_rate"
            ],
            "stressind_sys_stiffness_mla": self.df_sys.iloc[mla_pos_sys]["stiffness"],
            "stressind_sys_pc1_diastole_mla": self.df_sys.iloc[mla_pos_sys][
                "pc1_diastole"
            ],
            "stressind_sys_pc1_systole_mla": self.df_sys.iloc[mla_pos_sys][
                "pc1_systole"
            ],
            "ost_lumen_area_dia_rest": self.df_rest.iloc[ost_pos_rest]["lumen_area_dia"],
            "ost_lumen_area_sys_rest": self.df_rest.iloc[ost_pos_rest]["lumen_area_sys"],
            "ost_lumen_area_dia_stress": self.df_stress.iloc[ost_pos_stress]["lumen_area_dia"],
            "ost_lumen_area_sys_stress": self.df_stress.iloc[ost_pos_stress]["lumen_area_sys"],
            "ost_circ_dia_rest": self.df_rest.iloc[ost_pos_rest]["circ_dia"],
            "ost_circ_sys_rest": self.df_rest.iloc[ost_pos_rest]["circ_sys"],
            "ost_circ_dia_stress": self.df_stress.iloc[ost_pos_stress]["circ_dia"],
            "ost_circ_sys_stress": self.df_stress.iloc[ost_pos_stress]["circ_sys"],
            # "ost_lumen_narrow_dia_rest": 1 - (self.df_rest.iloc[ost_pos_rest]["lumen_area_dia"] / self.df_rest.iloc[0]["lumen_area_dia"]),
            # "ost_lumen_narrow_sys_rest": 1 - (self.df_rest.iloc[ost_pos_rest]["lumen_area_sys"] / self.df_rest.iloc[0]["lumen_area_sys"]),
            # "ost_lumen_narrow_dia_stress": 1 - (self.df_stress.iloc[ost_pos_stress]["lumen_area_dia"] / self.df_stress.iloc[0]["lumen_area_dia"]),
            # "ost_lumen_narrow_sys_stress": 1 - (self.df_stress.iloc[ost_pos_stress]["lumen_area_sys"] / self.df_stress.iloc[0]["lumen_area_sys"]),
            "mla_dia_rest": self.df_rest.iloc[mla_pos_rest]["lumen_area_dia"],
            "mla_sys_rest": self.df_rest.iloc[mla_pos_rest]["lumen_area_sys"],
            "mla_dia_stress": self.df_stress.iloc[mla_pos_stress]["lumen_area_dia"],
            "mla_sys_stress": self.df_stress.iloc[mla_pos_stress]["lumen_area_sys"],
            "mla_circ_dia_rest": self.df_rest.iloc[mla_pos_rest]["circ_dia"],
            "mla_circ_sys_rest": self.df_rest.iloc[mla_pos_rest]["circ_sys"],
            "mla_circ_dia_stress": self.df_stress.iloc[mla_pos_stress]["circ_dia"],
            "mla_circ_sys_stress": self.df_stress.iloc[mla_pos_stress]["circ_sys"],
            "mla_z_position_rest": self.df_rest.iloc[mla_pos_rest]["z_value"],
            "mla_z_position_stress": self.df_stress.iloc[mla_pos_stress]["z_value"],
            # "mln_dia_rest": 1 - (self.df_rest.iloc[ost_pos_rest]["lumen_area_dia"] / self.df_rest.iloc[0]["lumen_area_dia"]),
            # "mln_sys_rest": 1 - (self.df_rest.iloc[ost_pos_rest]["lumen_area_sys"] / self.df_rest.iloc[0]["lumen_area_sys"]),
            # "mln_dia_stress": 1 - (self.df_stress.iloc[ost_pos_stress]["lumen_area_dia"] / self.df_stress.iloc[0]["lumen_area_dia"]),
            # "mln_sys_stress": 1 - (self.df_stress.iloc[ost_pos_stress]["lumen_area_sys"] / self.df_stress.iloc[0]["lumen_area_sys"]),
            "ref_area_dia_rest": self.df_rest.iloc[ref_pos_rest]["lumen_area_dia"],
            "ref_area_sys_rest": self.df_rest.iloc[ref_pos_rest]["lumen_area_sys"],
            "ref_area_dia_stress": self.df_stress.iloc[ref_pos_stress]["lumen_area_dia"],
            "ref_area_sys_stress": self.df_stress.iloc[ref_pos_stress]["lumen_area_sys"],
            "ref_circ_dia_rest": self.df_rest.iloc[ref_pos_rest]["circ_dia"],
            "ref_circ_sys_rest": self.df_rest.iloc[ref_pos_rest]["circ_sys"],
            "ref_circ_dia_stress": self.df_stress.iloc[ref_pos_stress]["circ_dia"],
            "ref_circ_sys_stress": self.df_stress.iloc[ref_pos_stress]["circ_sys"],
            "ost_min_dist_dia_rest": self.df_rest.iloc[ost_pos_rest]["min_dist_dia"],
            "ost_min_dist_sys_rest": self.df_rest.iloc[ost_pos_rest]["min_dist_sys"],
            "ost_min_dist_dia_stress": self.df_stress.iloc[ost_pos_stress]["min_dist_dia"],
            "ost_min_dist_sys_stress": self.df_stress.iloc[ost_pos_stress]["min_dist_sys"],
            "mla_min_dist_dia_rest": self.df_rest.iloc[mla_pos_rest]["min_dist_dia"],
            "mla_min_dist_sys_rest": self.df_rest.iloc[mla_pos_rest]["min_dist_sys"],
            "mla_min_dist_dia_stress": self.df_stress.iloc[mla_pos_stress]["min_dist_dia"],
            "mla_min_dist_sys_stress": self.df_stress.iloc[mla_pos_stress]["min_dist_sys"],
            "ost_max_dist_dia_rest": self.df_rest.iloc[ost_pos_rest]["max_dist_dia"],
            "ost_max_dist_sys_rest": self.df_rest.iloc[ost_pos_rest]["max_dist_sys"],
            "ost_max_dist_dia_stress": self.df_stress.iloc[ost_pos_stress]["max_dist_dia"],
            "ost_max_dist_sys_stress": self.df_stress.iloc[ost_pos_stress]["max_dist_sys"],
            "mla_max_dist_dia_rest": self.df_rest.iloc[mla_pos_rest]["max_dist_dia"],
            "mla_max_dist_sys_rest": self.df_rest.iloc[mla_pos_rest]["max_dist_sys"],
            "mla_max_dist_dia_stress": self.df_stress.iloc[mla_pos_stress]["max_dist_dia"],
            "mla_max_dist_sys_stress": self.df_stress.iloc[mla_pos_stress]["max_dist_sys"],
        }

        # 7) Add section stats to the output DataFrame
        for key, value in section_stats.items():
            data[key] = value
        self.patient_df = data
        df_out = pd.DataFrame(data)

        # 8) Append or write out
        global_stats_path = os.path.join(self.global_dir, "local_patient_stats.csv")
        if os.path.exists(global_stats_path):
            df_existing = pd.read_csv(global_stats_path)
            df_existing = df_existing[df_existing["patient_id"] != self.patient_data.id]
            df_combined = pd.concat([df_existing, df_out], ignore_index=True)
            df_combined.to_csv(global_stats_path, index=False)
        else:
            df_out.to_csv(global_stats_path, index=False)

        logger.info(f"Saved patient stats to {global_stats_path}")
