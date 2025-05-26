import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from loguru import logger

from data_io.patient_data import PatientData
from stats.contour_measurements import compute_contour_properties


class PatientStats:
    def __init__(self, patient_data: PatientData, output_dir: str, global_dir: str):
        self.patient_data = patient_data
        self.output_dir = output_dir
        self.global_dir = global_dir
        self._create_output_dir()
        self.df_rest = pd.DataFrame()
        self.df_stress = pd.DataFrame()

    def _create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")

    def process_case(self):
        self.compute_lumen_changes()
        self.compute_sys_dia_properties(phase='rest')
        self.compute_sys_dia_properties(phase='stress')
    
    def _calculate_lumen_change(self, contours_dia, contours_sys, delta_pressure, delta_time, type='rest-stress'):
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

    def compute_diadia_syssys_properties(self, phase='dia-dia'):
        if phase == 'dia-dia':
            prop_rest = self.df_rest[['contour_id', 'z_value', 'lumen_area_dia', 'min_dist_dia', 'max_dist_dia', 'elliptic_ratio_dia']].values
            prop_stress = self.df_stress[['contour_id', 'z_value', 'lumen_area_dia', 'min_dist_dia', 'max_dist_dia', 'elliptic_ratio_dia']].values
            d_p = self.patient_data.pressure_stress[0] - self.patient_data.pressure_rest[0]
        elif phase == 'sys-sys':
            prop_rest = self.df_rest[['contour_id', 'z_value', 'lumen_area_sys', 'min_dist_sys', 'max_dist_sys', 'elliptic_ratio_sys']].values
            prop_stress = self.df_stress[['contour_id', 'z_value', 'lumen_area_sys', 'min_dist_sys', 'max_dist_sys', 'elliptic_ratio_sys']].values
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