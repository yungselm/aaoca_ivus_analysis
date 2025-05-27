import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from loguru import logger

from data_io.patient_data import PatientData
from stats.contour_measurements import compute_contour_properties

from typing import Tuple, List, Dict


class PatientStats:
    def __init__(self, patient_data: PatientData, output_dir: str, global_dir: str):
        self.patient_data = patient_data
        self.output_dir = output_dir
        self.global_dir = global_dir
        self._create_output_dir()
        self.df_rest = pd.DataFrame()
        self.df_stress = pd.DataFrame()

    def _create_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")

    def process_case(self) -> None:
        self.compute_lumen_changes()
        self.compute_sys_dia_properties(phase='rest')
        self.compute_sys_dia_properties(phase='stress')
        (_, stress_short_dia, stress_short_sys) = self.compute_lumen_changes_diadia_syssys()
        print(stress_short_dia, stress_short_sys)
        self.compute_diadia_syssys_properties(stress_short_dia, stress_short_sys, phase='dia-dia')
        self.compute_diadia_syssys_properties(stress_short_dia, stress_short_sys, phase='sys-sys')
    
    def _calculate_lumen_change(self, contours_dia, contours_sys, delta_pressure, delta_time=1, type='rest-stress') -> List[float]:
        """Calculate lumen change for each contour using PCA."""
        lumen_changes = []
        
        if contours_dia is None or contours_sys is None:
            logger.warning("Contours data is missing.")
            return lumen_changes

        # Extract unique contour IDs from both datasets
        dia_ids = np.unique(contours_dia[:, 0])
        sys_ids = np.unique(contours_sys[:, 0])

        # Verify matching contour IDs
        if not np.array_equal(dia_ids, sys_ids):
            logger.error("Mismatch in contour IDs between diastole and systole")
            return lumen_changes

        for contour_id in dia_ids:
            # Extract points for this contour (excluding ID column)
            dia_points = contours_dia[contours_dia[:, 0] == contour_id][:, 1:]
            sys_points = contours_sys[contours_sys[:, 0] == contour_id][:, 1:]

            # Verify point count (should be 501 points per contour)
            if len(dia_points) != 501 or len(sys_points) != 501:
                logger.warning(f"Contour {contour_id} has incorrect point count")
                continue

            # Calculate displacements (501 × 3 array)
            displacements = sys_points - dia_points

            # Perform PCA on the displacement vectors
            try:
                pca = PCA(n_components=3)
                pca.fit(displacements)
                total_variance = np.sum(pca.explained_variance_)
                if type == 'rest-stress':
                    lumen_change = total_variance / (delta_pressure * delta_time)
                elif type == 'dia-dia' or type == 'sys-sys':
                    lumen_change = total_variance / delta_pressure
                else:
                    logger.error(f"Invalid type: {type}. Use 'rest-stress', 'dia-dia', or 'sys-sys'.")
                    lumen_change = np.nan
                lumen_changes.append(lumen_change)
            except Exception as e:
                logger.error(f"PCA failed for contour {contour_id}: {str(e)}")
                lumen_changes.append(np.nan)

        return lumen_changes

    def compute_lumen_changes(self):
        """Main method to compute and save lumen changes for rest and stress."""
        results = {}
        
        # Process rest data
        if None not in self.patient_data.pressure_rest and None not in self.patient_data.time_rest:
            delta_p_rest = self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            delta_t_rest = self.patient_data.time_rest[1] - self.patient_data.time_rest[0]
            rest_changes = self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                self.patient_data.rest_contours_sys,
                delta_p_rest,
                delta_t_rest,
                type='rest-stress'
            )
            results['rest'] = rest_changes
        
        # Process stress data
        if None not in self.patient_data.pressure_stress and None not in self.patient_data.time_stress:
            delta_p_stress = self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]
            delta_t_stress = self.patient_data.time_stress[1] - self.patient_data.time_stress[0]
            stress_changes = self._calculate_lumen_change(
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
                delta_p_stress,
                delta_t_stress,
                type='rest-stress'
            )
            results['stress'] = stress_changes
        
        # Create and save DataFrames
        for condition in ['rest', 'stress']:
            if condition in results:
                df = pd.DataFrame({
                    'contour_id': range(len(results[condition])),
                    f'lumen_change_{condition}': results[condition]
                })
                output_path = os.path.join(self.output_dir, f'{condition}_lumen_changes.csv')
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {condition} lumen changes to {output_path}")

        return results

    def compute_sys_dia_properties(self, phase='rest'):
        if phase == 'rest':
            dia_prop = compute_contour_properties(self.patient_data.rest_contours_dia)
            sys_prop = compute_contour_properties(self.patient_data.rest_contours_sys)
            d_p = self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            d_t = self.patient_data.time_rest[1] - self.patient_data.time_rest[0]
        elif phase == 'stress':
            dia_prop = compute_contour_properties(self.patient_data.stress_contours_dia)
            sys_prop = compute_contour_properties(self.patient_data.stress_contours_sys)
            d_p = self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]
            d_t = self.patient_data.time_stress[1] - self.patient_data.time_stress[0]
        else:
            logger.error(f"Invalid phase: {phase}. Use 'rest' or 'stress'.")
            return

        df = pd.DataFrame({
            'contour_id': dia_prop[:, 0].astype(int),
            'z_value': dia_prop[:, 1],
            'delta_lumen_area': (sys_prop[:, 2] - dia_prop[:, 2]) / (d_p * d_t),
            'delta_min_dist': (sys_prop[:, 3] - dia_prop[:, 3]) / (d_p * d_t),
            'delta_max_dist': (sys_prop[:, 4] - dia_prop[:, 4]) / (d_p * d_t),
            'delta_elliptic_ratio': (sys_prop[:, 5] - dia_prop[:, 5]) / (d_p * d_t),
            'lumen_area_dia': dia_prop[:, 2],
            'lumen_area_sys': sys_prop[:, 2],
            'min_dist_dia': dia_prop[:, 3],
            'min_dist_sys': sys_prop[:, 3],
            'max_dist_dia': dia_prop[:, 4],
            'max_dist_sys': sys_prop[:, 4],
            'elliptic_ratio_dia': dia_prop[:, 5],
            'elliptic_ratio_sys': sys_prop[:, 5]
        })
        df['z_value'] = df['z_value'].apply(lambda x: abs(x - df['z_value'].max()))

        if phase == 'rest':
            lumen_changes_path = os.path.join(self.output_dir, 'rest_lumen_changes.csv')
        else:
            lumen_changes_path = os.path.join(self.output_dir, 'stress_lumen_changes.csv')

        if os.path.exists(lumen_changes_path):
            lumen_df = pd.read_csv(lumen_changes_path)
            df = df.merge(lumen_df, on='contour_id', how='left')
            df.to_csv(lumen_changes_path, index=False)
            if phase == 'rest':
                self.df_rest = df
            else:
                self.df_stress = df
        else:
            logger.warning(f"Lumen changes file not found: {lumen_changes_path}")

    def compute_lumen_changes_diadia_syssys(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """Main method to compute and save lumen changes for rest and stress."""
        stress_shortened_dia = self.shorten_and_reindex(
            self.patient_data.rest_contours_dia,
            self.patient_data.stress_contours_dia,
        )
        stress_shortened_sys = self.shorten_and_reindex(
            self.patient_data.rest_contours_sys,
            self.patient_data.stress_contours_sys,
        )

        results = {}
        
        # Process rest data
        if None not in self.patient_data.pressure_rest and None not in self.patient_data.pressure_stress:
            delta_p_rest = self.patient_data.pressure_stress[0] - self.patient_data.pressure_rest[0]
            rest_changes = self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                stress_shortened_dia,
                delta_p_rest,
                type='dia-dia'
            )
            results['diadia'] = rest_changes
        
        # Process stress data
        if None not in self.patient_data.pressure_rest and None not in self.patient_data.pressure_stress:
            delta_p_stress = self.patient_data.pressure_stress[1] - self.patient_data.pressure_rest[1]
            stress_changes = self._calculate_lumen_change(
                self.patient_data.rest_contours_sys,
                stress_shortened_sys,
                delta_p_stress,
                type='sys-sys'
            )
            results['syssys'] = stress_changes
        
        # Create and save DataFrames
        for condition in ['diadia', 'syssys']:
            if condition in results:
                df = pd.DataFrame({
                    'contour_id': range(len(results[condition])),
                    f'lumen_change_{condition}': results[condition]
                })
                output_path = os.path.join(self.output_dir, f'{condition}_lumen_changes.csv')
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {condition} lumen changes to {output_path}")
                cols = ['contour_id','x','y','z']
        
        stress_shortened_dia = pd.DataFrame(stress_shortened_dia, columns=cols)
        stress_shortened_sys = pd.DataFrame(stress_shortened_sys, columns=cols)

        return (results, stress_shortened_dia, stress_shortened_sys)

    @staticmethod
    def shorten_and_reindex(rest: np.ndarray, stress: np.ndarray, pts_per=501) -> np.ndarray:
        # 1) mean‐z per contour
        def mean_z(arr):
            ids = np.unique(arr[:,0])
            return {i: arr[arr[:,0]==i, 3].mean() for i in ids}

        rest_z   = mean_z(rest)
        stress_z = mean_z(stress)

        # 2) find ostium in stress
        stress_ids = sorted(stress_z.keys())
        ostium_id  = stress_ids[-1]

        # 3) match each rest contour→closest stress z (excluding ostium)
        matched = {}
        for rid, rz in rest_z.items():
            if rid == max(rest_z):
                continue
            candidates = {sid: abs(sz - rz)
                        for sid, sz in stress_z.items()
                        if sid != ostium_id}
            matched[rid] = min(candidates, key=candidates.get)

        # 4) select those + ostium
        sel_ids = list(matched.values()) + [ostium_id]
        short   = stress[np.isin(stress[:,0], sel_ids)]

        # 5) re‐index
        new_idx_map = {sid: i for i, sid in enumerate(matched.values())}
        new_idx_map[ostium_id] = len(matched)
        # vectorize mapping
        mapper = np.vectorize(lambda old: new_idx_map[int(old)])
        short[:,0] = mapper(short[:,0]).astype(int)

        return short

    def compute_diadia_syssys_properties(self, stress_shortened_dia, stress_shortened_sys, phase='dia-dia'):
        if phase == 'dia-dia':
            prop_rest = self.df_rest[['contour_id', 'z_value', 'lumen_area_dia', 'min_dist_dia', 'max_dist_dia', 'elliptic_ratio_dia']].values
            prop_stress = stress_shortened_dia[['contour_id', 'z_value', 'lumen_area_dia', 'min_dist_dia', 'max_dist_dia', 'elliptic_ratio_dia']].values
            d_p = self.patient_data.pressure_stress[0] - self.patient_data.pressure_rest[0]
        elif phase == 'sys-sys':
            prop_rest = self.df_rest[['contour_id', 'z_value', 'lumen_area_sys', 'min_dist_sys', 'max_dist_sys', 'elliptic_ratio_sys']].values
            prop_stress = stress_shortened_sys[['contour_id', 'z_value', 'lumen_area_sys', 'min_dist_sys', 'max_dist_sys', 'elliptic_ratio_sys']].values
            d_p = self.patient_data.pressure_stress[1] - self.patient_data.pressure_rest[1]
        else:
            logger.error(f"Invalid phase: {phase}. Use 'dia-dia' or 'sys-sys'.")
            return

        df = pd.DataFrame({
            'contour_id': prop_rest[:, 0].astype(int),
            'z_value': prop_rest[:, 1],
            'delta_lumen_area': (prop_stress[:, 2] - prop_rest[:, 2]) / d_p,
            'delta_min_dist': (prop_stress[:, 3] - prop_rest[:, 3]) / d_p,
            'delta_max_dist': (prop_stress[:, 4] - prop_rest[:, 4]) / d_p,
            'delta_elliptic_ratio': (prop_stress[:, 5] - prop_rest[:, 5]) / d_p,
            'lumen_area_dia': prop_rest[:, 2],
            'lumen_area_sys': prop_stress[:, 2],
            'min_dist_dia': prop_rest[:, 3],
            'min_dist_sys': prop_stress[:, 3],
            'max_dist_dia': prop_rest[:, 4],
            'max_dist_sys': prop_stress[:, 4],
            'elliptic_ratio_dia': prop_rest[:, 5],
            'elliptic_ratio_sys': prop_stress[:, 5]
        })
        df['z_value'] = df['z_value'].apply(lambda x: abs(x - df['z_value'].max()))

        if phase == 'rest':
            lumen_changes_path = os.path.join(self.output_dir, 'rest_lumen_changes.csv')
        else:
            lumen_changes_path = os.path.join(self.output_dir, 'stress_lumen_changes.csv')

        if os.path.exists(lumen_changes_path):
            lumen_df = pd.read_csv(lumen_changes_path)
            df = df.merge(lumen_df, on='contour_id', how='left')
            df.to_csv(lumen_changes_path, index=False)
            if phase == 'rest':
                self.df_rest = df
            else:
                self.df_stress = df
        else:
            logger.warning(f"Lumen changes file not found: {lumen_changes_path}")

    def save_selected_to_global_stats(self):
        pass