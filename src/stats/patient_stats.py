import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from loguru import logger

from data_io.patient_data import PatientData
from stats.contour_measurements import compute_contour_properties


class PatientStats:
    def __init__(self, patient_data: PatientData, output_dir: str):
        self.patient_data = patient_data
        self.output_dir = output_dir
        self._create_output_dir()

    def _create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")

    def process_case(self):
        pass

    def _calculate_lumen_change(self, contours_dia, contours_sys, delta_pressure, delta_time):
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
                lumen_change = total_variance * delta_pressure * delta_time
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
                delta_t_rest
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
                delta_t_stress
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
