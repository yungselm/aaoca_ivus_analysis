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
        self.output_dir   = os.path.join(output_dir, f"{patient_data.id}_stats")
        self.global_dir   = global_dir

        # — normalize EVERY input‐z once, up front —
        self._normalize_all_contour_z()

        self._create_output_dir()
        self.df_rest   = pd.DataFrame()
        self.df_stress = pd.DataFrame()
        self.df_dia = pd.DataFrame()
        self.df_sys = pd.DataFrame()

    def _normalize_all_contour_z(self) -> None:
        """
        For each of the four arrays in patient_data:
          rest_contours_dia, rest_contours_sys,
          stress_contours_dia, stress_contours_sys
        subtract its own max-z and take abs, in place.
        This ensures that ostium is at position 0.
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
        _, df_short_dia, df_short_sys = self.compute_lumen_changes_diadia_syssys()
        self.compute_diadia_syssys_properties(df_short_dia, phase='dia-dia')
        self.compute_diadia_syssys_properties(df_short_sys, phase='sys-sys')

        self.save_selected_to_global_stats()

    def compute_lumen_changes(self) -> Dict[str,List[float]]:
        results = {}
        # rest
        if None not in self.patient_data.pressure_rest and None not in self.patient_data.time_rest:
            dp = self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]
            dt = self.patient_data.time_rest[1]     - self.patient_data.time_rest[0]
            df = self._calculate_lumen_change(
                self.patient_data.rest_contours_dia,
                self.patient_data.rest_contours_sys,
                dp, dt, type='rest-stress'
            )
            results["rest"] = df
        # stress
        if None not in self.patient_data.pressure_stress and None not in self.patient_data.time_stress:
            dp = self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]
            dt = self.patient_data.time_stress[1]     - self.patient_data.time_stress[0]
            df = self._calculate_lumen_change(
                self.patient_data.stress_contours_dia,
                self.patient_data.stress_contours_sys,
                dp, dt, type='rest-stress'
            )
            results["stress"] = df

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
        type: str = 'rest-stress'
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
                pca = PCA(n_components=2)
                pca.fit(disp)
                pc_vars = pca.explained_variance_
                pc_ratio = pca.explained_variance_ratio_
                anisotropy = pc_vars[1] / pc_vars[0]
                pc1_dir, pc2_dir = pca.components_[:2]

                if type=='rest-stress':
                    change = pc_vars.sum() / (delta_pressure*delta_time)
                    anisotropy_index = anisotropy / (delta_pressure*delta_time)
                    pc1_var = pc_vars[0] / (delta_pressure*delta_time)
                    pc2_var = pc_vars[1] / (delta_pressure*delta_time)
                    pc_ratio = pc_ratio[0] / (delta_pressure*delta_time)
                    pc1_dir = pc1_dir / (delta_pressure*delta_time)
                    pc2_dir = pc2_dir / (delta_pressure*delta_time)
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

        df = pd.DataFrame({
            "contour_id":          range(len(lumen_changes)),
            "lumen_change":        lumen_changes,
            "anisotropy_index":    anisotropy_indices,
            "pc_ratio":            pc_ratios,
            "pc1_var":             pc1_vars,
            "pc2_var":             pc2_vars,
            "pc1_dir_x":           [d[0] for d in pc1_dirs],
            "pc1_dir_y":           [d[1] for d in pc1_dirs],
            "pc2_dir_x":           [d[0] for d in pc2_dirs],
            "pc2_dir_y":           [d[1] for d in pc2_dirs],
        })
        return df

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
    ) -> Dict[str,pd.DataFrame]:
        short_dia_rest, short_dia_stress = self.shorten_and_reindex(
            self.patient_data.rest_contours_dia,
            self.patient_data.stress_contours_dia
        )
        short_sys_rest, short_sys_stress = self.shorten_and_reindex(
            self.patient_data.rest_contours_sys,
            self.patient_data.stress_contours_sys
        )

        p_dia = compute_contour_properties(short_dia_stress)
        p_sys = compute_contour_properties(short_sys_stress)

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
                short_dia_rest,
                short_dia_stress,
                dp_dia,
                type='dia-dia'
            ),
            'syssys': self._calculate_lumen_change(
                short_sys_rest,
                short_sys_stress,
                dp_sys,
                type='sys-sys'
            )
        }

        for cond, df in results.items():
            # df = df.rename(columns={'lumen_change': f'lumen_change_{cond}'})
            path = os.path.join(self.output_dir, f'{cond}_lumen_changes.csv')
            df.to_csv(path, index=False)
            logger.info(f"Saved {cond} lumen changes to {path}")

        return results, df_short_dia, df_short_sys

    @staticmethod
    def shorten_and_reindex(rest: np.ndarray, stress: np.ndarray, pts_per: int = 501) -> Tuple[np.ndarray, np.ndarray]:
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
        
        # shorten the longer array by removing frames from the beginning
        # so that both arrays have the same number of unique contour_ids
        rest_ids = np.unique(rest[:, 0])
        short_ids = np.unique(short[:, 0])
        n_rest = len(rest_ids)
        n_short = len(short_ids)

        rest_new = None
        stress_new = None

        if n_short > n_rest:
            # Remove frames from the beginning of short to match rest
            ids_to_keep = short_ids[-n_rest:]
            short = short[np.isin(short[:, 0], ids_to_keep)]
            # Reindex contour_id to 0...n_rest-1
            new_map = {old: i for i, old in enumerate(sorted(ids_to_keep))}
            short[:, 0] = np.vectorize(lambda x: new_map[int(x)])(short[:, 0])
            rest[:, 0] = np.vectorize(lambda x: new_map[int(x)])(rest[:, 0])
            rest_new = rest
            stress_new = short
        elif n_rest > n_short:
            # Remove frames from the beginning of rest to match short
            ids_to_keep = rest_ids[-n_short:]
            rest = rest[np.isin(rest[:, 0], ids_to_keep)]
            # Reindex contour_id to 0...n_short-1
            new_map = {old: i for i, old in enumerate(sorted(ids_to_keep))}
            short[:, 0] = np.vectorize(lambda x: new_map[int(x)])(short[:, 0])
            rest[:, 0] = np.vectorize(lambda x: new_map[int(x)])(rest[:, 0])
            rest_new = rest
            stress_new = short
        else:
            rest_new = rest
            stress_new = short

        return (rest_new, stress_new)

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
        df_diadia = pd.read_csv(path)
        merged = df_diadia.merge(merged, on='contour_id', how='outer')
        merged.to_csv(path, index=False)
        if phase == 'dia-dia':
            self.df_dia = merged
        else:
            self.df_sys = merged
        logger.info(f"Saved {tag} properties to {path}")

    def save_selected_to_global_stats(self):
        """ Extensive summary functions for patients individual stats into a global dataframe.
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
                logger.warning(f"{self.patient_data.id}: no values >=1.3 for '{col}' in {phase_desc}; skipping stats")
                return None
            return hits[0]

        # 1) Determine ostium labels and their positional indices
        ost_id_rest   = self.df_rest['contour_id'].max()
        ost_id_stress = self.df_stress['contour_id'].max()
        ost_id_dia    = self.df_dia['contour_id'].max()
        ost_id_sys    = self.df_sys['contour_id'].max()

        try:
            ost_pos_rest   = self.df_rest.index[self.df_rest['contour_id'] == ost_id_rest][0]
            ost_pos_stress = self.df_stress.index[self.df_stress['contour_id'] == ost_id_stress][0]
            ost_pos_dia    = self.df_dia.index[self.df_dia['contour_id'] == ost_id_dia][0]
            ost_pos_sys    = self.df_sys.index[self.df_sys['contour_id'] == ost_id_sys][0]
        except IndexError as e:
            logger.error(f"{self.patient_data.id}: could not locate ostium row by label: {e}")
            return

        # 2) Find first-hit indices for elliptic_ratio >= 1.3 in each phase
        im_pos_rest   = None

        d_rest = first_hit(self.df_rest, 'elliptic_ratio_dia', 'rest-dia')
        s_rest = first_hit(self.df_rest, 'elliptic_ratio_sys', 'rest-sys')
        if d_rest is not None and s_rest is not None:
            im_pos_rest = (d_rest + s_rest) // 2
        elif d_rest is not None and s_rest is None:
            im_pos_rest = d_rest
        elif d_rest is None and s_rest is not None:
            im_pos_rest = s_rest
        else:
            return

        im_pos_stress = None
        d_str = first_hit(self.df_stress, 'elliptic_ratio_dia', 'stress-dia')
        s_str = first_hit(self.df_stress, 'elliptic_ratio_sys', 'stress-sys')
        if d_str is not None and s_str is not None:
            im_pos_stress = (d_str + s_str) // 2
        elif d_str is not None and s_str is None:
            im_pos_stress = d_str
        elif d_str is None and s_str is not None:
            im_pos_stress = s_str
        else:
            return

        im_pos_dia = None
        d_dia = first_hit(self.df_dia, 'elliptic_ratio_dia_rest', 'dia-rest')
        s_dia = first_hit(self.df_dia, 'elliptic_ratio_dia_stress', 'dia-stress')
        if d_dia is not None and s_dia is not None:
            im_pos_dia = (d_dia + s_dia) // 2
        elif d_dia is not None and s_dia is None:
            im_pos_dia = d_dia
        elif d_dia is None and s_dia is not None:
            im_pos_dia = s_dia
        else:
            return

        im_pos_sys = None
        d_sys = first_hit(self.df_sys, 'elliptic_ratio_sys_rest', 'sys-rest')
        s_sys = first_hit(self.df_sys, 'elliptic_ratio_sys_stress', 'sys-stress')
        if d_sys is not None and s_sys is not None:
            im_pos_sys = (d_sys + s_sys) // 2
        elif d_sys is not None and s_sys is None:
            im_pos_sys = d_sys
        elif d_sys is None and s_sys is not None:
            im_pos_sys = s_sys
        else:
            return

        # 3) Read z_value at these positions
        im_len_rest   = self.df_rest.iloc[im_pos_rest]['z_value']
        im_len_stress = self.df_stress.iloc[im_pos_stress]['z_value']
        im_len_dia    = (self.df_dia.iloc[im_pos_dia]['z_value_x'] + self.df_dia.iloc[im_pos_dia]['z_value_y']) / 2
        im_len_sys    = (self.df_sys.iloc[im_pos_sys]['z_value_x'] + self.df_sys.iloc[im_pos_sys]['z_value_y']) / 2

        # 4) Compute mean
        im_len = (im_len_rest + im_len_stress + im_len_dia + im_len_sys) / 4
        
        # needed for sectionwise
        im_20_per = im_len * 0.2
        n_intramural = round(im_len / im_20_per)
        n_segments = n_intramural + 3 # number of segments for extramural, 3 based on data
        print(f"patient {self.patient_data.id} with intramural length {im_len} has {n_intramural} intramural segments, {n_segments} total segments")

        section_stats = {}
        for phase, df in (
            ('pulsatile_rest', self.df_rest),
            ('pulsatile_stress', self.df_stress),
            ('stressind_dia', self.df_dia),
            ('stressind_sys', self.df_sys)
        ):
            for i in range(n_segments):
                z0 = i * im_20_per
                z1 = (i+1) * im_20_per
                if 'z_value' in df.columns:
                    sel = df[(df['z_value'] >= z0) & (df['z_value'] < z1)]
                else:
                    sel = df[(df['z_value_y'] >= z0) & (df['z_value_y'] < z1)]
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
                    mean_lumen = sel['delta_lumen_area'].mean()
                    mean_min_dist = sel['delta_min_dist'].mean()
                    mean_elliptic_ratio = sel['delta_elliptic_ratio'].mean()
                    mean_pca_glob = sel['lumen_change'].mean()
                    mean_pca_aniso = sel['anisotropy_index'].mean()
                    mean_pca_ratio = sel['pc_ratio'].mean()
                    mean_pca_pc1_var = sel['pc1_var'].mean()
                    mean_pca_pc2_var = sel['pc2_var'].mean()
                    mean_pca_pc1_dir_x = sel['pc1_dir_x'].mean()
                    mean_pca_pc1_dir_y = sel['pc1_dir_y'].mean()
                    mean_pca_pc2_dir_x = sel['pc2_dir_x'].mean()
                    mean_pca_pc2_dir_y = sel['pc2_dir_y'].mean()
                pct_label = (i+1) * 20
                section_stats[f'{phase}_pct_{pct_label}_lumen'] = mean_lumen
                section_stats[f'{phase}_pct_{pct_label}_min_dist'] = mean_min_dist
                section_stats[f'{phase}_pct_{pct_label}_ellip_ratio'] = mean_elliptic_ratio
                section_stats[f'{phase}_pct_{pct_label}_pca_glob'] = mean_pca_glob
                section_stats[f'{phase}_pct_{pct_label}_pca_aniso'] = mean_pca_aniso
                section_stats[f'{phase}_pct_{pct_label}_pca_ratio'] = mean_pca_ratio
                section_stats[f'{phase}_pct_{pct_label}_pca_pc1_var'] = mean_pca_pc1_var
                section_stats[f'{phase}_pct_{pct_label}_pca_pc2_var'] = mean_pca_pc2_var
                section_stats[f'{phase}_pct_{pct_label}_pca_pc1_dir_x'] = mean_pca_pc1_dir_x
                section_stats[f'{phase}_pct_{pct_label}_pca_pc1_dir_y'] = mean_pca_pc1_dir_y
                section_stats[f'{phase}_pct_{pct_label}_pca_pc2_dir_x'] = mean_pca_pc2_dir_x
                section_stats[f'{phase}_pct_{pct_label}_pca_pc2_dir_y'] = mean_pca_pc2_dir_y


        # 5) Compute MLA positions, clamped
        def clamp(idx, df):
            return max(0, min(idx, len(df) - 1))

        mla_pos_rest   = clamp((ost_pos_rest   - im_pos_rest)   // 2, self.df_rest)
        mla_pos_stress = clamp((ost_pos_stress - im_pos_stress) // 2, self.df_stress)
        mla_pos_dia    = clamp((ost_pos_dia    - im_pos_dia)    // 2, self.df_dia)
        mla_pos_sys    = clamp((ost_pos_sys    - im_pos_sys)    // 2, self.df_sys)

        # 6) Build the single-row DataFrame
        data = {
            'patient_id': [self.patient_data.id],
            'mean_intramural_length': [im_len],
            'pressure_change_pulsatile_rest': [self.patient_data.pressure_rest[1] - self.patient_data.pressure_rest[0]],
            'time_change_pulsatile_rest': [self.patient_data.time_rest[1] - self.patient_data.time_rest[0]],
            'pressure_change_pulsatile_stress': [self.patient_data.pressure_stress[1] - self.patient_data.pressure_stress[0]],
            'time_change_pulsatile_stress': [self.patient_data.time_stress[1] - self.patient_data.time_stress[0]],
            'pressure_change_stressind_dia': [self.patient_data.pressure_stress[0] - self.patient_data.pressure_rest[0]],
            'pressure_change_stressind_sys': [self.patient_data.pressure_stress[1] - self.patient_data.pressure_rest[1]],

            'pulsatile_rest_lumen_ost':   self.df_rest.iloc[ost_pos_rest]['delta_lumen_area'],
            'pulsatile_rest_min_ost':     self.df_rest.iloc[ost_pos_rest]['delta_min_dist'],
            'pulsatile_rest_ellip_ost':   self.df_rest.iloc[ost_pos_rest]['delta_elliptic_ratio'],
            'pulsatile_rest_pca_glob_ost':     self.df_rest.iloc[ost_pos_rest]['lumen_change'],
            'pulsatile_rest_pca_aniso_ost': self.df_rest.iloc[ost_pos_rest]['anisotropy_index'],
            'pulsatile_rest_pca_ratio_ost': self.df_rest.iloc[ost_pos_rest]['pc_ratio'],
            'pulsatile_rest_pca_pc1_var_ost': self.df_rest.iloc[ost_pos_rest]['pc1_var'],
            'pulsatile_rest_pca_pc2_var_ost': self.df_rest.iloc[ost_pos_rest]['pc2_var'],
            'pulsatile_rest_pca_pc1_dir_x_ost': self.df_rest.iloc[ost_pos_rest]['pc1_dir_x'],
            'pulsatile_rest_pca_pc1_dir_y_ost': self.df_rest.iloc[ost_pos_rest]['pc1_dir_y'],
            'pulsatile_rest_pca_pc2_dir_x_ost': self.df_rest.iloc[ost_pos_rest]['pc2_dir_x'],
            'pulsatile_rest_pca_pc2_dir_y_ost': self.df_rest.iloc[ost_pos_rest]['pc2_dir_y'],
            'pulsatile_rest_lumen_mla':   self.df_rest.iloc[mla_pos_rest]['delta_lumen_area'],
            'pulsatile_rest_min_mla':     self.df_rest.iloc[mla_pos_rest]['delta_min_dist'],
            'pulsatile_rest_ellip_mla':   self.df_rest.iloc[mla_pos_rest]['delta_elliptic_ratio'],
            'pulsatile_rest_pca_glob_mla':     self.df_rest.iloc[mla_pos_rest]['lumen_change'],
            'pulsatile_rest_pca_aniso_mla': self.df_rest.iloc[mla_pos_rest]['anisotropy_index'],
            'pulsatile_rest_pca_ratio_mla': self.df_rest.iloc[mla_pos_rest]['pc_ratio'],
            'pulsatile_rest_pca_pc1_var_mla': self.df_rest.iloc[mla_pos_rest]['pc1_var'],
            'pulsatile_rest_pca_pc2_var_mla': self.df_rest.iloc[mla_pos_rest]['pc2_var'],
            'pulsatile_rest_pca_pc1_dir_x_mla': self.df_rest.iloc[mla_pos_rest]['pc1_dir_x'],
            'pulsatile_rest_pca_pc1_dir_y_mla': self.df_rest.iloc[mla_pos_rest]['pc1_dir_y'],
            'pulsatile_rest_pca_pc2_dir_x_mla': self.df_rest.iloc[mla_pos_rest]['pc2_dir_x'],
            'pulsatile_rest_pca_pc2_dir_y_mla': self.df_rest.iloc[mla_pos_rest]['pc2_dir_y'],

            'pulsatile_stress_lumen_ost':   self.df_stress.iloc[ost_pos_stress]['delta_lumen_area'],
            'pulsatile_stress_min_ost':     self.df_stress.iloc[ost_pos_stress]['delta_min_dist'],
            'pulsatile_stress_ellip_ost':   self.df_stress.iloc[ost_pos_stress]['delta_elliptic_ratio'],
            'pulsatile_stress_pca_glob_ost':     self.df_stress.iloc[ost_pos_stress]['lumen_change'],
            'pulsatile_stress_pca_aniso_ost': self.df_stress.iloc[ost_pos_stress]['anisotropy_index'],
            'pulsatile_stress_pca_ratio_ost': self.df_stress.iloc[ost_pos_stress]['pc_ratio'],
            'pulsatile_stress_pca_pc1_var_ost': self.df_stress.iloc[ost_pos_stress]['pc1_var'],
            'pulsatile_stress_pca_pc2_var_ost': self.df_stress.iloc[ost_pos_stress]['pc2_var'],
            'pulsatile_stress_pca_pc1_dir_x_ost': self.df_stress.iloc[ost_pos_stress]['pc1_dir_x'],
            'pulsatile_stress_pca_pc1_dir_y_ost': self.df_stress.iloc[ost_pos_stress]['pc1_dir_y'],
            'pulsatile_stress_pca_pc2_dir_x_ost': self.df_stress.iloc[ost_pos_stress]['pc2_dir_x'],
            'pulsatile_stress_pca_pc2_dir_y_ost': self.df_stress.iloc[ost_pos_stress]['pc2_dir_y'],
            'pulsatile_stress_lumen_mla':   self.df_stress.iloc[mla_pos_stress]['delta_lumen_area'],
            'pulsatile_stress_min_mla':     self.df_stress.iloc[mla_pos_stress]['delta_min_dist'],
            'pulsatile_stress_ellip_mla':   self.df_stress.iloc[mla_pos_stress]['delta_elliptic_ratio'],
            'pulsatile_stress_pca_glob_mla':     self.df_stress.iloc[mla_pos_stress]['lumen_change'],
            'pulsatile_stress_pca_aniso_mla': self.df_stress.iloc[mla_pos_stress]['anisotropy_index'],
            'pulsatile_stress_pca_ratio_mla': self.df_stress.iloc[mla_pos_stress]['pc_ratio'],
            'pulsatile_stress_pca_pc1_var_mla': self.df_stress.iloc[mla_pos_stress]['pc1_var'],
            'pulsatile_stress_pca_pc2_var_mla': self.df_stress.iloc[mla_pos_stress]['pc2_var'],
            'pulsatile_stress_pca_pc1_dir_x_mla': self.df_stress.iloc[mla_pos_stress]['pc1_dir_x'],
            'pulsatile_stress_pca_pc1_dir_y_mla': self.df_stress.iloc[mla_pos_stress]['pc1_dir_y'],
            'pulsatile_stress_pca_pc2_dir_x_mla': self.df_stress.iloc[mla_pos_stress]['pc2_dir_x'],
            'pulsatile_stress_pca_pc2_dir_y_mla': self.df_stress.iloc[mla_pos_stress]['pc2_dir_y'],

            'stressind_dia_lumen_ost':   self.df_dia.iloc[ost_pos_dia]['delta_lumen_area'],
            'stressind_dia_min_ost':     self.df_dia.iloc[ost_pos_dia]['delta_min_dist'],
            'stressind_dia_ellip_ost':   self.df_dia.iloc[ost_pos_dia]['delta_elliptic_ratio'],
            'stressind_dia_pca_glob_ost':     self.df_dia.iloc[ost_pos_dia]['lumen_change'],
            'stressind_dia_pca_aniso_ost': self.df_dia.iloc[ost_pos_dia]['anisotropy_index'],
            'stressind_dia_pca_ratio_ost': self.df_dia.iloc[ost_pos_dia]['pc_ratio'],
            'stressind_dia_pca_pc1_var_ost': self.df_dia.iloc[ost_pos_dia]['pc1_var'],
            'stressind_dia_pca_pc2_var_ost': self.df_dia.iloc[ost_pos_dia]['pc2_var'],
            'stressind_dia_pca_pc1_dir_x_ost': self.df_dia.iloc[ost_pos_dia]['pc1_dir_x'],
            'stressind_dia_pca_pc1_dir_y_ost': self.df_dia.iloc[ost_pos_dia]['pc1_dir_y'],
            'stressind_dia_pca_pc2_dir_x_ost': self.df_dia.iloc[ost_pos_dia]['pc2_dir_x'],
            'stressind_dia_pca_pc2_dir_y_ost': self.df_dia.iloc[ost_pos_dia]['pc2_dir_y'],
            'stressind_dia_lumen_mla':   self.df_dia.iloc[mla_pos_dia]['delta_lumen_area'],
            'stressind_dia_min_mla':     self.df_dia.iloc[mla_pos_dia]['delta_min_dist'],
            'stressind_dia_ellip_mla':   self.df_dia.iloc[mla_pos_dia]['delta_elliptic_ratio'],
            'stressind_dia_pca_glob_mla':     self.df_dia.iloc[mla_pos_dia]['lumen_change'],
            'stressind_dia_pca_aniso_mla': self.df_dia.iloc[mla_pos_dia]['anisotropy_index'],
            'stressind_dia_pca_ratio_mla': self.df_dia.iloc[mla_pos_dia]['pc_ratio'],
            'stressind_dia_pca_pc1_var_mla': self.df_dia.iloc[mla_pos_dia]['pc1_var'],
            'stressind_dia_pca_pc2_var_mla': self.df_dia.iloc[mla_pos_dia]['pc2_var'],
            'stressind_dia_pca_pc1_dir_x_mla': self.df_dia.iloc[mla_pos_dia]['pc1_dir_x'],
            'stressind_dia_pca_pc1_dir_y_mla': self.df_dia.iloc[mla_pos_dia]['pc1_dir_y'],
            'stressind_dia_pca_pc2_dir_x_mla': self.df_dia.iloc[mla_pos_dia]['pc2_dir_x'],

            'stressind_sys_lumen_ost':   self.df_sys.iloc[ost_pos_sys]['delta_lumen_area'],
            'stressind_sys_min_ost':     self.df_sys.iloc[ost_pos_sys]['delta_min_dist'],
            'stressind_sys_ellip_ost':   self.df_sys.iloc[ost_pos_sys]['delta_elliptic_ratio'],
            'stressind_sys_pca_glob_ost':     self.df_sys.iloc[ost_pos_sys]['lumen_change'],
            'stressind_sys_pca_aniso_ost': self.df_sys.iloc[ost_pos_sys]['anisotropy_index'],
            'stressind_sys_pca_ratio_ost': self.df_sys.iloc[ost_pos_sys]['pc_ratio'],
            'stressind_sys_pca_pc1_var_ost': self.df_sys.iloc[ost_pos_sys]['pc1_var'],
            'stressind_sys_pca_pc2_var_ost': self.df_sys.iloc[ost_pos_sys]['pc2_var'],
            'stressind_sys_pca_pc1_dir_x_ost': self.df_sys.iloc[ost_pos_sys]['pc1_dir_x'],
            'stressind_sys_pca_pc1_dir_y_ost': self.df_sys.iloc[ost_pos_sys]['pc1_dir_y'],
            'stressind_sys_pca_pc2_dir_x_ost': self.df_sys.iloc[ost_pos_sys]['pc2_dir_x'],
            'stressind_sys_pca_pc2_dir_y_ost': self.df_sys.iloc[ost_pos_sys]['pc2_dir_y'],
            'stressind_sys_lumen_mla':   self.df_sys.iloc[mla_pos_sys]['delta_lumen_area'],
            'stressind_sys_min_mla':     self.df_sys.iloc[mla_pos_sys]['delta_min_dist'],
            'stressind_sys_ellip_mla':   self.df_sys.iloc[mla_pos_sys]['delta_elliptic_ratio'],
            'stressind_sys_pca_glob_mla':     self.df_sys.iloc[mla_pos_sys]['lumen_change'],
            'stressind_sys_pca_aniso_mla': self.df_sys.iloc[mla_pos_sys]['anisotropy_index'],
            'stressind_sys_pca_ratio_mla': self.df_sys.iloc[mla_pos_sys]['pc_ratio'],
            'stressind_sys_pca_pc1_var_mla': self.df_sys.iloc[mla_pos_sys]['pc1_var'],
            'stressind_sys_pca_pc2_var_mla': self.df_sys.iloc[mla_pos_sys]['pc2_var'],
            'stressind_sys_pca_pc1_dir_x_mla': self.df_sys.iloc[mla_pos_sys]['pc1_dir_x'],
            'stressind_sys_pca_pc1_dir_y_mla': self.df_sys.iloc[mla_pos_sys]['pc1_dir_y'],
            'stressind_sys_pca_pc2_dir_x_mla': self.df_sys.iloc[mla_pos_sys]['pc2_dir_x'],
            'stressind_sys_pca_pc2_dir_y_mla': self.df_sys.iloc[mla_pos_sys]['pc2_dir_y'],
        }

        # 7) Add section stats to the output DataFrame
        for key, value in section_stats.items():
            data[key] = value
        df_out = pd.DataFrame(data)

        # 8) Append or write out
        global_stats_path = os.path.join(self.global_dir, 'local_patient_stats.csv')
        if os.path.exists(global_stats_path):
            df_existing = pd.read_csv(global_stats_path)
            df_existing = df_existing[df_existing['patient_id'] != self.patient_data.id]
            df_combined = pd.concat([df_existing, df_out], ignore_index=True)
            df_combined.to_csv(global_stats_path, index=False)
        else:
            df_out.to_csv(global_stats_path, index=False)

        logger.info(f"Saved patient stats to {global_stats_path}")

