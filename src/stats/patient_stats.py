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
        self.output_dir   = output_dir
        self.global_dir   = global_dir

        # — normalize EVERY input‐z once, up front —
        self._normalize_all_contour_z()

        self._create_output_dir()
        self.df_rest   = pd.DataFrame()
        self.df_stress = pd.DataFrame()

    def _normalize_all_contour_z(self) -> None:
        """
        For each of the four arrays in patient_data:
          rest_contours_dia, rest_contours_sys,
          stress_contours_dia, stress_contours_sys
        subtract its own max-z and take abs, in place.
        """
        for attr in (
            'rest_contours_dia',
            'rest_contours_sys',
            'stress_contours_dia',
            'stress_contours_sys',
        ):
            arr = getattr(self.patient_data, attr, None)
            if isinstance(arr, np.ndarray) and arr.shape[1] > 3:
                z = arr[:, 3]
                arr[:, 3] = np.abs(z - z.max())
            else:
                logger.warning(f"Cannot normalize z for `{attr}`: not found or wrong shape")

    def _create_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")

    def process_case(self) -> None:
        # 1) rest-vs-stress
        self.compute_lumen_changes()
        self.compute_sys_dia_properties(phase='rest')
        self.compute_sys_dia_properties(phase='stress')

        # 2) diadia / syssys
        results, df_short_dia, df_short_sys = self.compute_lumen_changes_diadia_syssys()
        self.compute_diadia_syssys_properties(df_short_dia, phase='dia-dia')
        self.compute_diadia_syssys_properties(df_short_sys, phase='sys-sys')

    def _calculate_lumen_change(
        self,
        contours_dia: np.ndarray,
        contours_sys: np.ndarray,
        delta_pressure: float,
        delta_time: float = 1,
        type: str = 'rest-stress'
    ) -> List[float]:
        lumen_changes = []
        if contours_dia is None or contours_sys is None:
            logger.warning("Contours data is missing.")
            return lumen_changes

        dia_ids = np.unique(contours_dia[:,0])
        sys_ids = np.unique(contours_sys[:,0])
        if not np.array_equal(dia_ids, sys_ids):
            logger.error("Mismatch in contour IDs between diastole and systole")
            return lumen_changes

        for cid in dia_ids:
            d_pts = contours_dia [contours_dia [:,0]==cid][:,1:]
            s_pts = contours_sys[contours_sys[:,0]==cid][:,1:]
            if len(d_pts)!=501 or len(s_pts)!=501:
                logger.warning(f"Contour {cid} has incorrect point count")
                continue

            disp = s_pts - d_pts
            try:
                pca = PCA(n_components=3)
                pca.fit(disp)
                tot_var = pca.explained_variance_.sum()
                if type=='rest-stress':
                    change = tot_var/(delta_pressure*delta_time)
                else:  # dia–dia or sys–sys
                    change = tot_var/delta_pressure
                lumen_changes.append(change)
            except Exception as e:
                logger.error(f"PCA failed for contour {cid}: {e}")
                lumen_changes.append(np.nan)

        return lumen_changes

    def compute_lumen_changes(self) -> Dict[str,List[float]]:
        results = {}
        # rest
        if None not in self.patient_data.pressure_rest and None not in self.patient_data.time_rest:
            dp = self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            dt = self.patient_data.time_rest[1]     - self.patient_data.time_rest[0]
            results['rest'] = self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                self.patient_data.rest_contours_sys,
                dp, dt, type='rest-stress'
            )
        # stress
        if None not in self.patient_data.pressure_stress and None not in self.patient_data.time_stress:
            dp = self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]
            dt = self.patient_data.time_stress[1]     - self.patient_data.time_stress[0]
            results['stress'] = self._calculate_lumen_change(
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
                dp, dt, type='rest-stress'
            )

        for cond, changes in results.items():
            df = pd.DataFrame({
                'contour_id':           range(len(changes)),
                f'lumen_change_{cond}': changes
            })
            path = os.path.join(self.output_dir, f'{cond}_lumen_changes.csv')
            df.to_csv(path, index=False)
            logger.info(f"Saved {cond} lumen changes to {path}")
        return results

    def compute_sys_dia_properties(self, phase='rest') -> None:
        if phase=='rest':
            dia, sys = (self.patient_data.rest_contours_dia,
                        self.patient_data.rest_contours_sys)
            dp, dt   = (self.patient_data.pressure_rest[1]-self.patient_data.pressure_rest[0],
                        self.patient_data.time_rest[1]    -self.patient_data.time_rest[0])
        else:
            dia, sys = (self.patient_data.stress_contours_dia,
                        self.patient_data.stress_contours_sys)
            dp, dt   = (self.patient_data.pressure_stress[1]-self.patient_data.pressure_stress[0],
                        self.patient_data.time_stress[1]    -self.patient_data.time_stress[0])

        p_dia = compute_contour_properties(dia)
        p_sys = compute_contour_properties(sys)

        # p_dia[:,1] is already normalized z
        df = pd.DataFrame({
            'contour_id':         p_dia[:,0].astype(int),
            'z_value':            p_dia[:,1],
            'delta_lumen_area':   (p_sys[:,2] - p_dia[:,2])/(dp*dt),
            'delta_min_dist':     (p_sys[:,3] - p_dia[:,3])/(dp*dt),
            'delta_max_dist':     (p_sys[:,4] - p_dia[:,4])/(dp*dt),
            'delta_elliptic_ratio': (p_sys[:,5] - p_dia[:,5])/(dp*dt),
            'lumen_area_dia':     p_dia[:,2],
            'lumen_area_sys':     p_sys[:,2],
            'min_dist_dia':       p_dia[:,3],
            'min_dist_sys':       p_sys[:,3],
            'max_dist_dia':       p_dia[:,4],
            'max_dist_sys':       p_sys[:,4],
            'elliptic_ratio_dia': p_dia[:,5],
            'elliptic_ratio_sys': p_sys[:,5],
        })

        in_path = os.path.join(self.output_dir, f'{phase}_lumen_changes.csv')
        if os.path.exists(in_path):
            lum = pd.read_csv(in_path)
            df  = df.merge(lum, on='contour_id', how='left')
            df.to_csv(in_path, index=False)
            if phase=='rest':
                self.df_rest = df
            else:
                self.df_stress = df
        else:
            logger.warning(f"Missing file: {in_path}")

    def compute_lumen_changes_diadia_syssys(
        self
    ) -> Tuple[Dict[str,List[float]], pd.DataFrame, pd.DataFrame]:
        short_dia = self.shorten_and_reindex(
            self.patient_data.rest_contours_dia,
            self.patient_data.stress_contours_dia
        )
        short_sys = self.shorten_and_reindex(
            self.patient_data.rest_contours_sys,
            self.patient_data.stress_contours_sys
        )

        p_dia = compute_contour_properties(short_dia)
        p_sys = compute_contour_properties(short_sys)

        # p_dia[:,1] and p_sys[:,1] are already normalized z
        df_short_dia = pd.DataFrame({
            'contour_id':         p_dia[:,0].astype(int),
            'z_value':            p_dia[:,1],
            'lumen_area_dia':     p_dia[:,2],
            'min_dist_dia':       p_dia[:,3],
            'max_dist_dia':       p_dia[:,4],
            'elliptic_ratio_dia': p_dia[:,5],
        })
        df_short_sys = pd.DataFrame({
            'contour_id':         p_sys[:,0].astype(int),
            'z_value':            p_sys[:,1],
            'lumen_area_sys':     p_sys[:,2],
            'min_dist_sys':       p_sys[:,3],
            'max_dist_sys':       p_sys[:,4],
            'elliptic_ratio_sys': p_sys[:,5],
        })

        dp_dia = (self.patient_data.pressure_stress[0]
                  - self.patient_data.pressure_rest[0])
        dp_sys = (self.patient_data.pressure_stress[1]
                  - self.patient_data.pressure_rest[1])

        results = {
            'diadia': self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                short_dia,
                dp_dia,
                type='dia-dia'
            ),
            'syssys': self._calculate_lumen_change(
                self.patient_data.rest_contours_sys,
                short_sys,
                dp_sys,
                type='sys-sys'
            )
        }

        for cond, changes in results.items():
            df = pd.DataFrame({
                'contour_id':           range(len(changes)),
                f'lumen_change_{cond}': changes
            })
            path = os.path.join(self.output_dir, f'{cond}_lumen_changes.csv')
            df.to_csv(path, index=False)
            logger.info(f"Saved {cond} lumen changes to {path}")

        return results, df_short_dia, df_short_sys

    @staticmethod
    def shorten_and_reindex(rest: np.ndarray, stress: np.ndarray, pts_per: int = 501) -> np.ndarray:
        def mean_z(arr):
            ids = np.unique(arr[:,0])
            return {i: arr[arr[:,0]==i,3].mean() for i in ids}

        rest_z   = mean_z(rest)
        stress_z = mean_z(stress)
        ostium   = max(stress_z.keys())

        matched = {}
        for rid, rz in rest_z.items():
            if rid == max(rest_z):
                continue
            candidates = {sid: abs(sz - rz) for sid,sz in stress_z.items() if sid != ostium}
            matched[rid] = min(candidates, key=candidates.get)

        sel = list(matched.values()) + [ostium]
        short = stress[np.isin(stress[:,0], sel)]

        new_map = {sid:i for i,sid in enumerate(matched.values())}
        new_map[ostium] = len(matched)
        mapper = np.vectorize(lambda old:new_map[int(old)])
        short[:,0] = mapper(short[:,0]).astype(int)
        return short

    def compute_diadia_syssys_properties(
        self,
        df_short: pd.DataFrame,
        phase: str = 'dia-dia'
    ) -> None:
        if phase=='dia-dia':
            cols = ['lumen_area_dia','min_dist_dia','max_dist_dia','elliptic_ratio_dia']
            dp   = (self.patient_data.pressure_stress[0]
                    - self.patient_data.pressure_rest[0])
        else:
            cols = ['lumen_area_sys','min_dist_sys','max_dist_sys','elliptic_ratio_sys']
            dp   = (self.patient_data.pressure_stress[1]
                    - self.patient_data.pressure_rest[1])

        rest_df = self.df_rest[['contour_id','z_value'] + cols].copy()
        rest_df.rename(columns={c:f"{c}_rest" for c in cols}, inplace=True)

        stress_df = df_short[['contour_id','z_value'] + cols].copy()
        stress_df.rename(columns={c:f"{c}_stress" for c in cols}, inplace=True)

        merged = rest_df.merge(stress_df, on='contour_id')
        base   = phase.split('-')[0]  # 'dia' or 'sys'

        merged['delta_lumen_area']     = (merged[f'lumen_area_{base}_stress']
                                         - merged[f'lumen_area_{base}_rest']) / dp
        merged['delta_min_dist']       = (merged[f'min_dist_{base}_stress']
                                         - merged[f'min_dist_{base}_rest']) / dp
        merged['delta_max_dist']       = (merged[f'max_dist_{base}_stress']
                                         - merged[f'max_dist_{base}_rest']) / dp
        merged['delta_elliptic_ratio'] = (merged[f'elliptic_ratio_{base}_stress']
                                         - merged[f'elliptic_ratio_{base}_rest']) / dp

        tag  = 'diadia' if phase=='dia-dia' else 'syssys'
        path = os.path.join(self.output_dir, f'{tag}_lumen_changes.csv')
        merged.to_csv(path, index=False)
        logger.info(f"Saved {tag} properties to {path}")

    def save_selected_to_global_stats(self):
        pass
