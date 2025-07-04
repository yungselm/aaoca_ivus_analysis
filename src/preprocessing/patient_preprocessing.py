from __future__ import annotations

import os
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from data_io.patient_data import PatientData
from loguru import logger
from preprocessing.contour_measurements import calculate_displacement_map
from preprocessing.contour_measurements import calculate_measurement_map
from preprocessing.contour_measurements import compute_contour_properties
from advanced_visualizations.plot_frame_diff import plot_frame_diff
from sklearn.decomposition import PCA


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
        # rest
        if (
            None not in self.patient_data.pressure_rest
            and None not in self.patient_data.time_rest
        ):
            dp = self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            dt = self.patient_data.time_rest[1] - self.patient_data.time_rest[0]
            df = self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                self.patient_data.rest_contours_sys,
                dp,
                dt,
                type="rest-stress",
            )
            results["rest"] = df
        # stress
        if (
            None not in self.patient_data.pressure_stress
            and None not in self.patient_data.time_stress
        ):
            dp = (
                self.patient_data.pressure_stress[1]
                - self.patient_data.pressure_stress[0]
            )
            dt = self.patient_data.time_stress[1] - self.patient_data.time_stress[0]
            df = self._calculate_lumen_change(
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
                dp,
                dt,
                type="rest-stress",
            )
            results["stress"] = df

        # dia-dia
        if (
            None not in self.patient_data.pressure_rest
            and None not in self.patient_data.pressure_stress
        ):
            dp = (
                self.patient_data.pressure_stress[0]
                - self.patient_data.pressure_rest[0]
            )
            df = self._calculate_lumen_change(
                self.patient_data.dia_contours_rest,
                self.patient_data.dia_contours_stress,
                dp,
                type="dia-dia",
            )
            results["dia_dia"] = df

        # sys-sys
        if (
            None not in self.patient_data.pressure_rest
            and None not in self.patient_data.pressure_stress
        ):
            dp = (
                self.patient_data.pressure_stress[1]
                - self.patient_data.pressure_rest[1]
            )
            df = self._calculate_lumen_change(
                self.patient_data.sys_contours_rest,
                self.patient_data.sys_contours_stress,
                dp,
                type="sys-sys",
            )
            results["sys_sys"] = df

        for cond, df in results.items():
            # df = df.rename(columns={'lumen_change': f'lumen_change_{cond}'})
            path = os.path.join(self.output_dir, f"{cond}_lumen_changes.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved {cond} lumen changes to {path}")
        return results

    def _calculate_lumen_change(
        self,
        contours_dia: np.ndarray,
        contours_sys: np.ndarray,
        delta_pressure: float,
        delta_time: float = 1,
        type: str = "rest-stress",
    ) -> List[float]:
        lumen_changes = []
        anisotropy_indices = []
        pc_ratios = []
        pc1_vars = []
        pc2_vars = []
        pc1_dirs = []
        pc2_dirs = []

        if contours_dia is None or contours_sys is None:
            logger.warning("Contours data is missing.")
            return lumen_changes

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
            return lumen_changes

        for cid in dia_ids:
            d_pts = contours_dia[contours_dia[:, 0] == cid][:, 1:]
            s_pts = contours_sys[contours_sys[:, 0] == cid][:, 1:]
            if len(d_pts) != 501 or len(s_pts) != 501:
                logger.warning(f"Contour {cid} has incorrect point count")
                continue

            disp = s_pts - d_pts
            try:
                pca = PCA(n_components=2)
                pca.fit(disp)
                pc_vars = pca.explained_variance_
                pc_ratio = pca.explained_variance_ratio_
                anisotropy = pc_vars[1] / pc_vars[0]
                pc1_dir, pc2_dir = pca.components_[:2]

                if type == "rest-stress":
                    change = pc_vars.sum() / (delta_pressure * delta_time)
                    anisotropy_index = anisotropy / (delta_pressure * delta_time)
                    pc1_var = pc_vars[0] / (delta_pressure * delta_time)
                    pc2_var = pc_vars[1] / (delta_pressure * delta_time)
                    pc_ratio = pc_ratio[0] / (delta_pressure * delta_time)
                    pc1_dir = pc1_dir / (delta_pressure * delta_time)
                    pc2_dir = pc2_dir / (delta_pressure * delta_time)
                else:  # dia–dia or sys–sys
                    change = pc_vars.sum() / delta_pressure
                    anisotropy_index = anisotropy / delta_pressure
                    pc1_var = pc_vars[0] / delta_pressure
                    pc2_var = pc_vars[1] / delta_pressure
                    pc_ratio = pc_ratio[0] / delta_pressure
                    pc1_dir = pc1_dir / delta_pressure
                    pc2_dir = pc2_dir / delta_pressure
                lumen_changes.append(change)
                anisotropy_indices.append(anisotropy_index)
                pc_ratios.append(pc_ratio)
                pc1_vars.append(pc1_var)
                pc2_vars.append(pc2_var)
                pc1_dirs.append(pc1_dir)
                pc2_dirs.append(pc2_dir)

            except Exception as e:
                logger.error(f"PCA failed for contour {cid}: {e}")
                lumen_changes.append(np.nan)
                anisotropy_indices.append(np.nan)
                pc_ratios.append(np.nan)
                pc1_vars.append(np.nan)
                pc2_vars.append(np.nan)
                pc1_dirs.append(np.nan)
                pc2_dirs.append(np.nan)

        df = pd.DataFrame(
            {
                "contour_id": range(len(lumen_changes)),
                "lumen_change": lumen_changes,
                "anisotropy_index": anisotropy_indices,
                "pc_ratio": pc_ratios,
                "pc1_var": pc1_vars,
                "pc2_var": pc2_vars,
                "pc1_dir_x": [d[0] for d in pc1_dirs],
                "pc1_dir_y": [d[1] for d in pc1_dirs],
                "pc2_dir_x": [d[0] for d in pc2_dirs],
                "pc2_dir_y": [d[1] for d in pc2_dirs],
            }
        )
        return df

    def compute_sys_dia_properties(self, phase="rest") -> None:
        if phase == "rest":
            dia, sys = (
                self.patient_data.rest_contours_dia,
                self.patient_data.rest_contours_sys,
            )
            dp, dt = (
                self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0],
                self.patient_data.time_rest[1] - self.patient_data.time_rest[0],
            )
        elif phase == "stress":
            dia, sys = (
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
            )
            dp, dt = (
                self.patient_data.pressure_stress[1]
                - self.patient_data.pressure_stress[0],
                self.patient_data.time_stress[1] - self.patient_data.time_stress[0],
            )
        elif phase == "dia_dia":
            dia, sys = (
                self.patient_data.dia_contours_rest,
                self.patient_data.dia_contours_stress,
            )
            dp, dt = (
                self.patient_data.pressure_stress[0]
                - self.patient_data.pressure_rest[0],
                1, # No time change for dia-dia
            )
        elif phase == "sys_sys":
            dia, sys = (
                self.patient_data.sys_contours_rest,
                self.patient_data.sys_contours_stress,
            )
            dp, dt = (
                self.patient_data.pressure_stress[1]
                - self.patient_data.pressure_rest[1],
                1, # No time change for sys-sys
            )
        else:
            logger.error(f"Invalid phase: {phase}, expected 'rest', 'stress', 'dia_dia' or 'sys_sys'")

        p_dia = compute_contour_properties(dia)
        p_sys = compute_contour_properties(sys)

        # p_dia[:,1] is already normalized z
        if phase == "rest" or phase == "stress":
            df = pd.DataFrame(
                {
                    "contour_id": p_dia[:, 0].astype(int),
                    "z_value": p_dia[:, 1],
                    "delta_lumen_area": (p_sys[:, 2] - p_dia[:, 2]) / (dp * dt),
                    "delta_min_dist": (p_sys[:, 3] - p_dia[:, 3]) / (dp * dt),
                    "delta_max_dist": (p_sys[:, 4] - p_dia[:, 4]) / (dp * dt),
                    "delta_elliptic_ratio": (p_sys[:, 5] - p_dia[:, 5]) / (dp * dt),
                    "lumen_area_dia": p_dia[:, 2],
                    "lumen_area_sys": p_sys[:, 2],
                    "min_dist_dia": p_dia[:, 3],
                    "min_dist_sys": p_sys[:, 3],
                    "max_dist_dia": p_dia[:, 4],
                    "max_dist_sys": p_sys[:, 4],
                    "elliptic_ratio_dia": p_dia[:, 5],
                    "elliptic_ratio_sys": p_sys[:, 5],
                }
            )
        elif phase == "dia_dia":
            df = pd.DataFrame(
                {
                    "contour_id": p_dia[:, 0].astype(int),
                    "z_value": p_dia[:, 1],
                    "delta_lumen_area": (p_sys[:, 2] - p_dia[:, 2]) / (dp * dt),
                    "delta_min_dist": (p_sys[:, 3] - p_dia[:, 3]) / (dp * dt),
                    "delta_max_dist": (p_sys[:, 4] - p_dia[:, 4]) / (dp * dt),
                    "delta_elliptic_ratio": (p_sys[:, 5] - p_dia[:, 5]) / (dp * dt),
                    "lumen_area_dia_rest": p_dia[:, 2],
                    "lumen_area_dia_stress": p_sys[:, 2],
                    "min_dist_dia_rest": p_dia[:, 3],
                    "min_dist_dia_stress": p_sys[:, 3],
                    "max_dist_dia_rest": p_dia[:, 4],
                    "max_dist_dia_stress": p_sys[:, 4],
                    "elliptic_ratio_dia_rest": p_dia[:, 5],
                    "elliptic_ratio_dia_stress": p_sys[:, 5],
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "contour_id": p_dia[:, 0].astype(int),
                    "z_value": p_dia[:, 1],
                    "delta_lumen_area": (p_sys[:, 2] - p_dia[:, 2]) / (dp * dt),
                    "delta_min_dist": (p_sys[:, 3] - p_dia[:, 3]) / (dp * dt),
                    "delta_max_dist": (p_sys[:, 4] - p_dia[:, 4]) / (dp * dt),
                    "delta_elliptic_ratio": (p_sys[:, 5] - p_dia[:, 5]) / (dp * dt),
                    "lumen_area_sys_rest": p_dia[:, 2],
                    "lumen_area_sys_stress": p_sys[:, 2],
                    "min_dist_sys_rest": p_dia[:, 3],
                    "min_dist_sys_stress": p_sys[:, 3],
                    "max_dist_sys_rest": p_dia[:, 4],
                    "max_dist_sys_stress": p_sys[:, 4],
                    "elliptic_ratio_sys_rest": p_dia[:, 5],
                    "elliptic_ratio_sys_stress": p_sys[:, 5],
                }
            )            

        in_path = os.path.join(self.output_dir, f"{phase}_lumen_changes.csv")
        if os.path.exists(in_path):
            lum = pd.read_csv(in_path)
            df = df.merge(lum, on="contour_id", how="left")
            df.to_csv(in_path, index=False)
            if phase == "rest":
                self.df_rest = df
            elif phase == "stress":
                self.df_stress = df
            elif phase == "dia_dia":
                self.df_dia = df
            elif phase == "sys_sys":
                self.df_sys = df
            else:
                logger.error(f"Invalid phase: {phase}, expected 'rest', 'stress', 'dia_dia' or 'sys_sys'")
        else:
            logger.warning(f"Missing file: {in_path}")

    def save_selected_to_global_stats(self):
        """Extensive summary functions for patients individual stats into a global dataframe.
            Column definition are explicit, allows for easy updates.
            Input:
            - self.df_rest, self.df_stress, self.df_dia, self.df_sys
        - Output:
            - DataFrame with selected statistics, saved to global stats directory.
        """

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
        im_len = (im_len_rest + im_len_stress + im_len_dia + im_len_sys) / 4

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
                    mean_lumen = np.nan
                    mean_min_dist = np.nan
                    mean_elliptic_ratio = np.nan
                    mean_pca_glob = np.nan
                    mean_pca_aniso = np.nan
                    mean_pca_ratio = np.nan
                    mean_pca_pc1_var = np.nan
                    mean_pca_pc2_var = np.nan
                    mean_pca_pc1_dir_x = np.nan
                    mean_pca_pc1_dir_y = np.nan
                    mean_pca_pc2_dir_x = np.nan
                    mean_pca_pc2_dir_y = np.nan
                else:
                    mean_lumen = sel["delta_lumen_area"].mean()
                    mean_min_dist = sel["delta_min_dist"].mean()
                    mean_elliptic_ratio = sel["delta_elliptic_ratio"].mean()
                    mean_pca_glob = sel["lumen_change"].mean()
                    mean_pca_aniso = sel["anisotropy_index"].mean()
                    mean_pca_ratio = sel["pc_ratio"].mean()
                    mean_pca_pc1_var = sel["pc1_var"].mean()
                    mean_pca_pc2_var = sel["pc2_var"].mean()
                    mean_pca_pc1_dir_x = sel["pc1_dir_x"].mean()
                    mean_pca_pc1_dir_y = sel["pc1_dir_y"].mean()
                    mean_pca_pc2_dir_x = sel["pc2_dir_x"].mean()
                    mean_pca_pc2_dir_y = sel["pc2_dir_y"].mean()
                pct_label = (i + 1) * 20
                section_stats[f"{phase}_pct_{pct_label}_lumen"] = mean_lumen
                section_stats[f"{phase}_pct_{pct_label}_min_dist"] = mean_min_dist
                section_stats[f"{phase}_pct_{pct_label}_ellip_ratio"] = (
                    mean_elliptic_ratio
                )
                section_stats[f"{phase}_pct_{pct_label}_pca_glob"] = mean_pca_glob
                section_stats[f"{phase}_pct_{pct_label}_pca_aniso"] = mean_pca_aniso
                section_stats[f"{phase}_pct_{pct_label}_pca_ratio"] = mean_pca_ratio
                section_stats[f"{phase}_pct_{pct_label}_pca_pc1_var"] = mean_pca_pc1_var
                section_stats[f"{phase}_pct_{pct_label}_pca_pc2_var"] = mean_pca_pc2_var
                section_stats[f"{phase}_pct_{pct_label}_pca_pc1_dir_x"] = (
                    mean_pca_pc1_dir_x
                )
                section_stats[f"{phase}_pct_{pct_label}_pca_pc1_dir_y"] = (
                    mean_pca_pc1_dir_y
                )
                section_stats[f"{phase}_pct_{pct_label}_pca_pc2_dir_x"] = (
                    mean_pca_pc2_dir_x
                )
                section_stats[f"{phase}_pct_{pct_label}_pca_pc2_dir_y"] = (
                    mean_pca_pc2_dir_y
                )

        # 5) Compute MLA positions, clamped
        def clamp(idx, df):
            return max(0, min(idx, len(df) - 1))

        mla_pos_rest = clamp((ost_pos_rest - im_pos_rest) // 2, self.df_rest)
        mla_pos_stress = clamp((ost_pos_stress - im_pos_stress) // 2, self.df_stress)
        mla_pos_dia = clamp((ost_pos_dia - im_pos_dia) // 2, self.df_dia)
        mla_pos_sys = clamp((ost_pos_sys - im_pos_sys) // 2, self.df_sys)

        # 6) Build the single-row DataFrame
        data = {
            "patient_id": [self.patient_data.id],
            "mean_intramural_length": [im_len],
            "pressure_change_pulsatile_rest": [
                self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            ],
            "time_change_pulsatile_rest": [
                self.patient_data.time_rest[1] - self.patient_data.time_rest[0]
            ],
            "pressure_change_pulsatile_stress": [
                self.patient_data.pressure_stress[1]
                - self.patient_data.pressure_stress[0]
            ],
            "time_change_pulsatile_stress": [
                self.patient_data.time_stress[1] - self.patient_data.time_stress[0]
            ],
            "pressure_change_stressind_dia": [
                self.patient_data.pressure_stress[0]
                - self.patient_data.pressure_rest[0]
            ],
            "pressure_change_stressind_sys": [
                self.patient_data.pressure_stress[1]
                - self.patient_data.pressure_rest[1]
            ],
            "pulsatile_rest_lumen_ost": self.df_rest.iloc[ost_pos_rest][
                "delta_lumen_area"
            ],
            "pulsatile_rest_min_ost": self.df_rest.iloc[ost_pos_rest]["delta_min_dist"],
            "pulsatile_rest_ellip_ost": self.df_rest.iloc[ost_pos_rest][
                "delta_elliptic_ratio"
            ],
            "pulsatile_rest_pca_glob_ost": self.df_rest.iloc[ost_pos_rest][
                "lumen_change"
            ],
            "pulsatile_rest_pca_aniso_ost": self.df_rest.iloc[ost_pos_rest][
                "anisotropy_index"
            ],
            "pulsatile_rest_pca_ratio_ost": self.df_rest.iloc[ost_pos_rest]["pc_ratio"],
            "pulsatile_rest_pca_pc1_var_ost": self.df_rest.iloc[ost_pos_rest][
                "pc1_var"
            ],
            "pulsatile_rest_pca_pc2_var_ost": self.df_rest.iloc[ost_pos_rest][
                "pc2_var"
            ],
            "pulsatile_rest_pca_pc1_dir_x_ost": self.df_rest.iloc[ost_pos_rest][
                "pc1_dir_x"
            ],
            "pulsatile_rest_pca_pc1_dir_y_ost": self.df_rest.iloc[ost_pos_rest][
                "pc1_dir_y"
            ],
            "pulsatile_rest_pca_pc2_dir_x_ost": self.df_rest.iloc[ost_pos_rest][
                "pc2_dir_x"
            ],
            "pulsatile_rest_pca_pc2_dir_y_ost": self.df_rest.iloc[ost_pos_rest][
                "pc2_dir_y"
            ],
            "pulsatile_rest_lumen_mla": self.df_rest.iloc[mla_pos_rest][
                "delta_lumen_area"
            ],
            "pulsatile_rest_min_mla": self.df_rest.iloc[mla_pos_rest]["delta_min_dist"],
            "pulsatile_rest_ellip_mla": self.df_rest.iloc[mla_pos_rest][
                "delta_elliptic_ratio"
            ],
            "pulsatile_rest_pca_glob_mla": self.df_rest.iloc[mla_pos_rest][
                "lumen_change"
            ],
            "pulsatile_rest_pca_aniso_mla": self.df_rest.iloc[mla_pos_rest][
                "anisotropy_index"
            ],
            "pulsatile_rest_pca_ratio_mla": self.df_rest.iloc[mla_pos_rest]["pc_ratio"],
            "pulsatile_rest_pca_pc1_var_mla": self.df_rest.iloc[mla_pos_rest][
                "pc1_var"
            ],
            "pulsatile_rest_pca_pc2_var_mla": self.df_rest.iloc[mla_pos_rest][
                "pc2_var"
            ],
            "pulsatile_rest_pca_pc1_dir_x_mla": self.df_rest.iloc[mla_pos_rest][
                "pc1_dir_x"
            ],
            "pulsatile_rest_pca_pc1_dir_y_mla": self.df_rest.iloc[mla_pos_rest][
                "pc1_dir_y"
            ],
            "pulsatile_rest_pca_pc2_dir_x_mla": self.df_rest.iloc[mla_pos_rest][
                "pc2_dir_x"
            ],
            "pulsatile_rest_pca_pc2_dir_y_mla": self.df_rest.iloc[mla_pos_rest][
                "pc2_dir_y"
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
            "pulsatile_stress_pca_glob_ost": self.df_stress.iloc[ost_pos_stress][
                "lumen_change"
            ],
            "pulsatile_stress_pca_aniso_ost": self.df_stress.iloc[ost_pos_stress][
                "anisotropy_index"
            ],
            "pulsatile_stress_pca_ratio_ost": self.df_stress.iloc[ost_pos_stress][
                "pc_ratio"
            ],
            "pulsatile_stress_pca_pc1_var_ost": self.df_stress.iloc[ost_pos_stress][
                "pc1_var"
            ],
            "pulsatile_stress_pca_pc2_var_ost": self.df_stress.iloc[ost_pos_stress][
                "pc2_var"
            ],
            "pulsatile_stress_pca_pc1_dir_x_ost": self.df_stress.iloc[ost_pos_stress][
                "pc1_dir_x"
            ],
            "pulsatile_stress_pca_pc1_dir_y_ost": self.df_stress.iloc[ost_pos_stress][
                "pc1_dir_y"
            ],
            "pulsatile_stress_pca_pc2_dir_x_ost": self.df_stress.iloc[ost_pos_stress][
                "pc2_dir_x"
            ],
            "pulsatile_stress_pca_pc2_dir_y_ost": self.df_stress.iloc[ost_pos_stress][
                "pc2_dir_y"
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
            "pulsatile_stress_pca_glob_mla": self.df_stress.iloc[mla_pos_stress][
                "lumen_change"
            ],
            "pulsatile_stress_pca_aniso_mla": self.df_stress.iloc[mla_pos_stress][
                "anisotropy_index"
            ],
            "pulsatile_stress_pca_ratio_mla": self.df_stress.iloc[mla_pos_stress][
                "pc_ratio"
            ],
            "pulsatile_stress_pca_pc1_var_mla": self.df_stress.iloc[mla_pos_stress][
                "pc1_var"
            ],
            "pulsatile_stress_pca_pc2_var_mla": self.df_stress.iloc[mla_pos_stress][
                "pc2_var"
            ],
            "pulsatile_stress_pca_pc1_dir_x_mla": self.df_stress.iloc[mla_pos_stress][
                "pc1_dir_x"
            ],
            "pulsatile_stress_pca_pc1_dir_y_mla": self.df_stress.iloc[mla_pos_stress][
                "pc1_dir_y"
            ],
            "pulsatile_stress_pca_pc2_dir_x_mla": self.df_stress.iloc[mla_pos_stress][
                "pc2_dir_x"
            ],
            "pulsatile_stress_pca_pc2_dir_y_mla": self.df_stress.iloc[mla_pos_stress][
                "pc2_dir_y"
            ],
            "stressind_dia_lumen_ost": self.df_dia.iloc[ost_pos_dia][
                "delta_lumen_area"
            ],
            "stressind_dia_min_ost": self.df_dia.iloc[ost_pos_dia]["delta_min_dist"],
            "stressind_dia_ellip_ost": self.df_dia.iloc[ost_pos_dia][
                "delta_elliptic_ratio"
            ],
            "stressind_dia_pca_glob_ost": self.df_dia.iloc[ost_pos_dia]["lumen_change"],
            "stressind_dia_pca_aniso_ost": self.df_dia.iloc[ost_pos_dia][
                "anisotropy_index"
            ],
            "stressind_dia_pca_ratio_ost": self.df_dia.iloc[ost_pos_dia]["pc_ratio"],
            "stressind_dia_pca_pc1_var_ost": self.df_dia.iloc[ost_pos_dia]["pc1_var"],
            "stressind_dia_pca_pc2_var_ost": self.df_dia.iloc[ost_pos_dia]["pc2_var"],
            "stressind_dia_pca_pc1_dir_x_ost": self.df_dia.iloc[ost_pos_dia][
                "pc1_dir_x"
            ],
            "stressind_dia_pca_pc1_dir_y_ost": self.df_dia.iloc[ost_pos_dia][
                "pc1_dir_y"
            ],
            "stressind_dia_pca_pc2_dir_x_ost": self.df_dia.iloc[ost_pos_dia][
                "pc2_dir_x"
            ],
            "stressind_dia_pca_pc2_dir_y_ost": self.df_dia.iloc[ost_pos_dia][
                "pc2_dir_y"
            ],
            "stressind_dia_lumen_mla": self.df_dia.iloc[mla_pos_dia][
                "delta_lumen_area"
            ],
            "stressind_dia_min_mla": self.df_dia.iloc[mla_pos_dia]["delta_min_dist"],
            "stressind_dia_ellip_mla": self.df_dia.iloc[mla_pos_dia][
                "delta_elliptic_ratio"
            ],
            "stressind_dia_pca_glob_mla": self.df_dia.iloc[mla_pos_dia]["lumen_change"],
            "stressind_dia_pca_aniso_mla": self.df_dia.iloc[mla_pos_dia][
                "anisotropy_index"
            ],
            "stressind_dia_pca_ratio_mla": self.df_dia.iloc[mla_pos_dia]["pc_ratio"],
            "stressind_dia_pca_pc1_var_mla": self.df_dia.iloc[mla_pos_dia]["pc1_var"],
            "stressind_dia_pca_pc2_var_mla": self.df_dia.iloc[mla_pos_dia]["pc2_var"],
            "stressind_dia_pca_pc1_dir_x_mla": self.df_dia.iloc[mla_pos_dia][
                "pc1_dir_x"
            ],
            "stressind_dia_pca_pc1_dir_y_mla": self.df_dia.iloc[mla_pos_dia][
                "pc1_dir_y"
            ],
            "stressind_dia_pca_pc2_dir_x_mla": self.df_dia.iloc[mla_pos_dia][
                "pc2_dir_x"
            ],
            "stressind_sys_lumen_ost": self.df_sys.iloc[ost_pos_sys][
                "delta_lumen_area"
            ],
            "stressind_sys_min_ost": self.df_sys.iloc[ost_pos_sys]["delta_min_dist"],
            "stressind_sys_ellip_ost": self.df_sys.iloc[ost_pos_sys][
                "delta_elliptic_ratio"
            ],
            "stressind_sys_pca_glob_ost": self.df_sys.iloc[ost_pos_sys]["lumen_change"],
            "stressind_sys_pca_aniso_ost": self.df_sys.iloc[ost_pos_sys][
                "anisotropy_index"
            ],
            "stressind_sys_pca_ratio_ost": self.df_sys.iloc[ost_pos_sys]["pc_ratio"],
            "stressind_sys_pca_pc1_var_ost": self.df_sys.iloc[ost_pos_sys]["pc1_var"],
            "stressind_sys_pca_pc2_var_ost": self.df_sys.iloc[ost_pos_sys]["pc2_var"],
            "stressind_sys_pca_pc1_dir_x_ost": self.df_sys.iloc[ost_pos_sys][
                "pc1_dir_x"
            ],
            "stressind_sys_pca_pc1_dir_y_ost": self.df_sys.iloc[ost_pos_sys][
                "pc1_dir_y"
            ],
            "stressind_sys_pca_pc2_dir_x_ost": self.df_sys.iloc[ost_pos_sys][
                "pc2_dir_x"
            ],
            "stressind_sys_pca_pc2_dir_y_ost": self.df_sys.iloc[ost_pos_sys][
                "pc2_dir_y"
            ],
            "stressind_sys_lumen_mla": self.df_sys.iloc[mla_pos_sys][
                "delta_lumen_area"
            ],
            "stressind_sys_min_mla": self.df_sys.iloc[mla_pos_sys]["delta_min_dist"],
            "stressind_sys_ellip_mla": self.df_sys.iloc[mla_pos_sys][
                "delta_elliptic_ratio"
            ],
            "stressind_sys_pca_glob_mla": self.df_sys.iloc[mla_pos_sys]["lumen_change"],
            "stressind_sys_pca_aniso_mla": self.df_sys.iloc[mla_pos_sys][
                "anisotropy_index"
            ],
            "stressind_sys_pca_ratio_mla": self.df_sys.iloc[mla_pos_sys]["pc_ratio"],
            "stressind_sys_pca_pc1_var_mla": self.df_sys.iloc[mla_pos_sys]["pc1_var"],
            "stressind_sys_pca_pc2_var_mla": self.df_sys.iloc[mla_pos_sys]["pc2_var"],
            "stressind_sys_pca_pc1_dir_x_mla": self.df_sys.iloc[mla_pos_sys][
                "pc1_dir_x"
            ],
            "stressind_sys_pca_pc1_dir_y_mla": self.df_sys.iloc[mla_pos_sys][
                "pc1_dir_y"
            ],
            "stressind_sys_pca_pc2_dir_x_mla": self.df_sys.iloc[mla_pos_sys][
                "pc2_dir_x"
            ],
            "stressind_sys_pca_pc2_dir_y_mla": self.df_sys.iloc[mla_pos_sys][
                "pc2_dir_y"
            ],
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
