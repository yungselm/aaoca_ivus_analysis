from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from scipy.stats import ranksums
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import levene

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


class PatientStats:
    def __init__(self, input_dir: str, output_path: str):
        self.input_dir = input_dir
        self.output_path = output_path
        self.data = self.load_patient_data()

    def run(self):
        ffr_df = self.pairwise_pressure_split("ffr_pos")
        ifr_df = self.pairwise_pressure_split("ifr_pos")
        print(ffr_df)
        print(ifr_df)
        self.run_linear_pairwise()
        path_data = os.path.join(self.input_dir, "test_data.csv")
        self.data.to_csv(path_data)
        self.univariate_logistic_regression()
        self.univariate_logistic_regression("iFR_mean_dobu")
        self.kmeans_clustering(3)

    def load_patient_data(self) -> pd.DataFrame:
        """
        Load patient data from input path.
        """
        try:
            path_ivus = os.path.join(self.input_dir, "local_patient_stats.csv")
            path_pressure = os.path.join(self.input_dir, "global_data.csv")
            data_ivus = pd.read_csv(path_ivus)
            data_pressure = pd.read_csv(path_pressure)
            logger.info(f"Loaded patient data from {self.input_dir}")
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")

        valid_patient_ids = set(data_ivus["patient_id"])
        data_pressure = data_pressure[
            data_pressure["patient_id"].isin(valid_patient_ids)
        ]

        exclude_prefixes = (
            "ostial_",
            "mla_",
            "global_",
            "reference_",
            "wallthickness_",
            "narco_id",
        )
        cols_to_keep = [
            col for col in data_pressure.columns if not col.startswith(exclude_prefixes)
        ]
        data_pressure = data_pressure[cols_to_keep]

        data = pd.merge(data_ivus, data_pressure, on="patient_id", how="inner")
        data["ffr_pos"] = np.where(data["pdpa_mean_dobu"] <= 0.8, 1, 0)
        data["ifr_pos"] = np.where(data["iFR_mean_dobu"] <= 0.8, 1, 0)

        return data

    def pairwise_pressure_split(
        self,
        split_var: str = "ffr_pos",
        min_n: int = 6,
        save: bool = True,
        apply_fdr: bool = True,
        out_dir: str | None = None,
    ):
        """
        Compare numeric metrics between groups defined by `split_var` (expects binary 0/1).
        - split_var: column name containing 0/1 to split dataset
        - min_n: minimum samples per group to run a comparison
        - save: whether to save CSV to disk
        - apply_fdr: whether to apply Benjamini-Hochberg correction across tested metrics
        - out_dir: override output directory (defaults to self.output_path)
        """

        # ---- validations ----
        if split_var not in self.data.columns:
            logger.error(f"Split variable '{split_var}' not found in dataset")
            return

        # Ensure groups are 0/1; treat any non-zero as 1
        self.data["_split_bin"] = self.data[split_var].apply(
            lambda x: 1 if x == 1 else 0
        )

        group_pos = self.data[self.data["_split_bin"] == 1]
        group_neg = self.data[self.data["_split_bin"] == 0]

        logger.info(
            f"Group sizes - {split_var}=1: {len(group_pos)}, {split_var}=0: {len(group_neg)}"
        )

        # ---- build metric list: numeric only, exclude IDs/flags ----
        exclude = {"patient_id", "ffr_pos", "ifr_pos", split_var, "_split_bin"}
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        metrics = [c for c in numeric_cols if c not in exclude]

        results = []

        # helper: BH correction later
        def bh_adjust(pvals):
            """Benjamini-Hochberg FDR (returns adjusted p-values in original order)."""
            pvals = np.array(pvals)
            n = len(pvals)
            order = np.argsort(pvals)
            ranked = np.empty(n, dtype=float)
            cummin = 1.0
            for i, idx in enumerate(order[::-1], start=1):
                rank = n - i + 1
                adj = (pvals[idx] * n) / rank
                cummin = min(cummin, adj)
                ranked[idx] = cummin
            # cap at 1.0
            ranked = np.minimum(ranked, 1.0)
            return ranked

        for metric in metrics:
            # coerce to numeric and drop NA
            data_pos = pd.to_numeric(group_pos[metric], errors="coerce").dropna()
            data_neg = pd.to_numeric(group_neg[metric], errors="coerce").dropna()

            n_pos = len(data_pos)
            n_neg = len(data_neg)

            if n_pos < min_n or n_neg < min_n:
                logger.debug(
                    f"Skipping {metric}: insufficient samples ({split_var}=1:{n_pos}, {split_var}=0:{n_neg})"
                )
                continue

            # Normality (Shapiro) — only run if sample size is within shapiro's valid range;
            # wrap in try/except to be robust
            normal_pos = normal_neg = False
            try:
                if 3 <= n_pos <= 5000:
                    _, p_sh_pos = shapiro(data_pos)
                    normal_pos = p_sh_pos > 0.05
                else:
                    # too small or large for a reliable Shapiro result; be conservative and skip normal assumption
                    normal_pos = False

                if 3 <= n_neg <= 5000:
                    _, p_sh_neg = shapiro(data_neg)
                    normal_neg = p_sh_neg > 0.05
                else:
                    normal_neg = False
            except Exception as e:
                logger.warning(f"Shapiro normality test failed for {metric}: {e}")
                normal_pos = normal_neg = False

            # Choose test:
            if normal_pos and normal_neg:
                # check variance equality
                try:
                    _, p_levene = levene(data_pos, data_neg)
                    equal_var = p_levene > 0.05
                except Exception:
                    equal_var = False
                stat, pval = ttest_ind(data_pos, data_neg, equal_var=equal_var)
                test_used = "t-test"
                # Cohen's d (pooled)
                pooled_sd = np.sqrt(
                    (
                        (n_pos - 1) * np.var(data_pos, ddof=1)
                        + (n_neg - 1) * np.var(data_neg, ddof=1)
                    )
                    / (n_pos + n_neg - 2)
                )
                effect_size = (
                    (np.mean(data_pos) - np.mean(data_neg)) / pooled_sd
                    if pooled_sd != 0
                    else np.nan
                )
            else:
                # non-parametric
                stat, pval = ranksums(
                    data_pos, data_neg
                )  # returns z-statistic and p-value
                test_used = "ranksums"
                z_stat = stat
                effect_size = z_stat / np.sqrt(n_pos + n_neg)  # r-like effect size

            # summary stats: medians & MAD (robust)
            pos_median = np.median(data_pos)
            neg_median = np.median(data_neg)
            pos_mad = np.median(np.abs(data_pos - pos_median))
            neg_mad = np.median(np.abs(data_neg - neg_median))

            results.append(
                {
                    "metric": metric,
                    f"{split_var}_1_n": n_pos,
                    f"{split_var}_1_median": pos_median,
                    f"{split_var}_1_mad": pos_mad,
                    f"{split_var}_0_n": n_neg,
                    f"{split_var}_0_median": neg_median,
                    f"{split_var}_0_mad": neg_mad,
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "test": test_used,
                    "effect_size": (
                        float(effect_size) if np.isfinite(effect_size) else np.nan
                    ),
                }
            )

        # assemble dataframe
        results_df = pd.DataFrame(results)

        if results_df.empty:
            logger.warning("No metrics passed filters; no results generated.")
            # cleanup helper column
            self.data.drop(columns=["_split_bin"], inplace=True, errors="ignore")
            return

        # multiple testing correction (optional)
        if apply_fdr and "p_value" in results_df.columns:
            results_df["p_adj"] = bh_adjust(results_df["p_value"].values)
            results_df["significant_fdr"] = results_df["p_adj"] < 0.05
        else:
            results_df["p_adj"] = results_df["p_value"]
            results_df["significant_fdr"] = results_df["p_adj"] < 0.05

        # persist into the instance and optionally to disk
        attr_name = f"{split_var}_comparison"
        setattr(self, attr_name, results_df)

        if save:
            out_dir = out_dir or getattr(self, "output_path", ".")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{split_var}_comparison.csv")
            results_df.to_csv(out_path, index=False)
            logger.success(f"Saved {len(results_df)} comparisons to {out_path}")

        # cleanup helper column
        self.data.drop(columns=["_split_bin"], inplace=True, errors="ignore")

        return results_df

    def run_linear_pairwise(self):
        # --- outcomes (the stressind variables you originally asked about) ---
        outcomes = [
            "stressind_dia_lumen_percent_ost",
            "stressind_dia_min_percent_ost",
            "stressind_sys_lumen_percent_ost",
            "stressind_sys_min_percent_ost",
            "stressind_dia_lumen_percent_mla",
            "stressind_dia_min_percent_mla",
            "stressind_sys_lumen_percent_mla",
            "stressind_sys_min_percent_mla",
        ]

        # --- predictors (long list you provided) ---
        predictors = [
            "mean_intramural_length",
            "pressure_change_pulsatile_rest",
            "time_change_pulsatile_rest",
            "pressure_change_pulsatile_stress",
            "time_change_pulsatile_stress",
            "pressure_change_stressind_dia",
            "pressure_change_stressind_sys",
            "pulsatile_rest_lumen_ost",
            "pulsatile_rest_min_ost",
            "pulsatile_rest_ellip_ost",
            "pulsatile_rest_stretch_ost",
            "pulsatile_rest_stretch_rate_ost",
            "pulsatile_rest_stiffness_ost",
            "pulsatile_rest_pc1_diastole_ost",
            "pulsatile_rest_pct_20_min_dist",
            "pulsatile_rest_pct_20_ellip_ratio",
            "pulsatile_rest_pct_20_stretch",
            "pulsatile_rest_pct_20_stretch_rate",
            "pulsatile_rest_pct_20_stiffness",
            "pulsatile_rest_pct_20_pc1_dia",
            "pulsatile_rest_pct_20_pc1_sys",
            "pulsatile_rest_pct_40_lumen",
            "pulsatile_rest_pct_40_min_dist",
            "pulsatile_rest_pct_40_ellip_ratio",
            "pulsatile_rest_pct_40_stretch",
            "pulsatile_rest_pct_40_stretch_rate",
            "pulsatile_rest_pct_40_stiffness",
            "pulsatile_rest_pct_40_pc1_dia",
            "pulsatile_rest_pct_40_pc1_sys",
            "pulsatile_rest_pct_60_lumen",
            "pulsatile_rest_pct_60_min_dist",
            "pulsatile_rest_pct_60_ellip_ratio",
            "pulsatile_rest_pct_60_stretch",
            "pulsatile_rest_pct_60_stretch_rate",
            "pulsatile_rest_pct_60_stiffness",
            "pulsatile_rest_pct_60_pc1_dia",
            "pulsatile_rest_pct_60_pc1_sys",
            "pulsatile_rest_pct_80_lumen",
            "pulsatile_rest_pct_80_min_dist",
            "pulsatile_rest_pct_80_ellip_ratio",
            "pulsatile_rest_pct_80_stretch",
            "pulsatile_rest_pct_80_stretch_rate",
            "pulsatile_rest_pct_80_stiffness",
            "pulsatile_rest_pct_80_pc1_dia",
            "pulsatile_rest_pct_80_pc1_sys",
            "pulsatile_rest_pct_100_lumen",
            "pulsatile_rest_pct_100_min_dist",
            "pulsatile_rest_pct_100_ellip_ratio",
            "pulsatile_rest_pct_100_stretch",
            "pulsatile_rest_pct_100_stretch_rate",
            "pulsatile_rest_pct_100_stiffness",
            "pulsatile_rest_pct_100_pc1_dia",
            "pulsatile_rest_pct_100_pc1_sys",
            "pulsatile_rest_pct_120_lumen",
            "pulsatile_rest_pct_120_min_dist",
            "pulsatile_rest_pct_120_ellip_ratio",
            "pulsatile_rest_pct_120_stretch",
            "pulsatile_rest_pct_120_stretch_rate",
            "pulsatile_rest_pct_120_stiffness",
            "pulsatile_rest_pct_120_pc1_dia",
            "pulsatile_rest_pct_120_pc1_sys",
            "pulsatile_rest_pct_140_lumen",
            "pulsatile_rest_pct_140_min_dist",
            "pulsatile_rest_pct_140_ellip_ratio",
            "pulsatile_rest_pct_140_stretch",
            "pulsatile_rest_pct_140_stretch_rate",
            "pulsatile_rest_pct_140_stiffness",
            "pulsatile_rest_pct_140_pc1_dia",
            "pulsatile_rest_pct_140_pc1_sys",
            "pulsatile_rest_pct_160_lumen",
            "pulsatile_rest_pct_160_min_dist",
            "pulsatile_rest_pct_160_ellip_ratio",
            "pulsatile_rest_pct_160_stretch",
            "pulsatile_rest_pct_160_stretch_rate",
            "pulsatile_rest_pct_160_stiffness",
            "pulsatile_rest_pct_160_pc1_dia",
            "pulsatile_rest_pct_160_pc1_sys",
            "pulsatile_rest_pc1_systole_ost",
            "pulsatile_rest_lumen_mla",
            "pulsatile_rest_min_mla",
            "pulsatile_rest_ellip_mla",
            "pulsatile_rest_stretch_mla",
            "pulsatile_rest_stretch_rate_mla",
            "pulsatile_rest_stiffness_mla",
            "pulsatile_rest_pc1_diastole_mla",
            "pulsatile_rest_pc1_systole_mla",
        ]

        # Keep only predictors actually in the dataframe
        predictors = [p for p in predictors if p in self.data.columns]
        if not predictors:
            logger.error("No predictor columns found in data. Check names.")
            return

        all_results = []

        for outcome in outcomes:
            if outcome not in self.data.columns:
                logger.warning(f"Outcome '{outcome}' not found in data. Skipping.")
                continue

            rows = []
            # we'll compute BH-FDR per outcome at the end, collect p-values
            pvals = []
            for predictor in predictors:
                # Prepare a minimal dataframe with the two columns
                sub = self.data[[predictor, outcome]].copy()

                # Drop if predictor has no variance
                if sub[predictor].nunique(dropna=True) <= 1:
                    logger.debug(
                        f"Predictor {predictor} has <=1 unique value; skipping."
                    )
                    continue

                # Simple mean imputation for missing values
                imp = SimpleImputer(strategy="mean")
                arr = imp.fit_transform(sub)

                X = arr[:, 0]
                y = arr[:, 1]
                # build design matrix with intercept
                X_design = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X_design).fit()
                except Exception as e:
                    logger.exception(
                        f"Regression failed for {predictor} -> {outcome}: {e}"
                    )
                    continue

                coef = model.params[1] if len(model.params) > 1 else np.nan
                se = model.bse[1] if len(model.bse) > 1 else np.nan
                tval = model.tvalues[1] if len(model.tvalues) > 1 else np.nan
                pval = model.pvalues[1] if len(model.pvalues) > 1 else np.nan
                n_obs = int(
                    np.sum(~np.isnan(self.data[[predictor, outcome]]).any(axis=1))
                )
                row = {
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": n_obs,
                    "coef": float(coef),
                    "coef_se": float(se),
                    "t": float(tval),
                    "pvalue": float(pval),
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                }
                rows.append(row)
                pvals.append(pval)

            # adjust p-values (Benjamini-Hochberg) for this outcome
            if rows:
                pvals_arr = np.array([r["pvalue"] for r in rows])
                # multipletests returns: rej, pvals_corrected, _, _
                rej, pvals_bh, _, _ = multipletests(
                    pvals_arr, alpha=0.05, method="fdr_bh"
                )
                for i, r in enumerate(rows):
                    r["pvalue_adj_fdr_bh"] = float(pvals_bh[i])
                    r["significant_fdr_bh"] = bool(rej[i])

                # save per-outcome CSV
                df_out = pd.DataFrame(rows).sort_values("pvalue")
                fname = os.path.join(
                    self.output_path, f"{outcome}_linear_regression_results.csv"
                )
                df_out.to_csv(fname, index=False)
                logger.info(f"Wrote {fname} with {len(df_out)} predictor results.")

                all_results.extend(rows)

        # save combined CSV
        if all_results:
            df_all = pd.DataFrame(all_results).sort_values(["outcome", "pvalue"])
            all_path = os.path.join(self.output_path, "all_linear_results.csv")
            df_all.to_csv(all_path, index=False)
            logger.info(f"Wrote combined results to {all_path}.")
        else:
            logger.warning("No regression results to save.")

    def process_local_patient_data(self) -> pd.DataFrame:
        pass

    def exploratory_data_analysis(self):
        pass

    # def kmeans_clustering(
    #     self,
    #     n_clusters: int | None = None,
    #     k_search_range=(2, 6),
    #     plot_path: str | None = None,
    # ):
    #     """
    #     KMeans clustering on pulsatile_rest features excluding pct_* and numeric-percentage slots (20/40/...).
    #     - n_clusters: if provided, use that. If None, pick by silhouette in k_search_range.
    #     - plot_path: if provided, save the PCA scatter to that path; otherwise show it.
    #     """

    #     # sanity
    #     if self.data is None or self.data.empty:
    #         raise ValueError("No patient data loaded. Run load_patient_data first.")

    #     # --- safe pc1 delta computation (creates pc1_ost_change, pc1_ost_rel_change, pc1_mla_change, pc1_mla_rel_change) ---
    #     def safe_compute_delta(df, s_col, d_col, out_prefix):
    #         if s_col in df.columns and d_col in df.columns:
    #             delta = df[s_col] - df[d_col]
    #             rel = np.where(df[d_col].abs() > 1e-12, delta / df[d_col], np.nan)
    #             df[f"{out_prefix}_change"] = delta
    #             df[f"{out_prefix}_rel_change"] = rel
    #         else:
    #             logger.debug(
    #                 f"Skipping delta creation for missing columns: {s_col}, {d_col}"
    #             )

    #     safe_compute_delta(
    #         self.data,
    #         "pulsatile_rest_pc1_systole_ost",
    #         "pulsatile_rest_pc1_diastole_ost",
    #         "pc1_ost",
    #     )
    #     safe_compute_delta(
    #         self.data,
    #         "pulsatile_rest_pc1_systole_mla",
    #         "pulsatile_rest_pc1_diastole_mla",
    #         "pc1_mla",
    #     )

    #     # --- pick features: only pulsatile_rest_ features + the two pressure/time change cols, but exclude pct_* and _20_, _40_, etc. ---
    #     allowed_prefixes = ("pulsatile_rest_",)

    #     # substrings to exclude (the pct groups and numeric percent markers)
    #     exclude_substrings = (
    #         "_pct_",
    #         "_20_",
    #         "_40_",
    #         "_60_",
    #         "_80_",
    #         "_100_",
    #         "_120_",
    #         "_140_",
    #         "_160_",
    #     )

    #     def keep_col(col: str) -> bool:
    #         # starts with allowed prefixes AND does not contain any exclude substring
    #         if not col.startswith(allowed_prefixes):
    #             return False
    #         for ex in exclude_substrings:
    #             if ex in col:
    #                 return False
    #         return True

    #     rest_features = [col for col in self.data.columns if keep_col(col)]

    #     # include pc1 features if they were created
    #     pc1_features = [
    #         c
    #         for c in (
    #             "pc1_ost_change",
    #             "pc1_ost_rel_change",
    #             "pc1_mla_change",
    #             "pc1_mla_rel_change",
    #         )
    #         if c in self.data.columns
    #     ]

    #     features = rest_features + pc1_features

    #     if not features:
    #         raise ValueError(
    #             "No matching features found for clustering. Check column names / prefixes."
    #         )

    #     print("Using features for clustering (count={}):".format(len(features)))
    #     print(features)

    #     # --- build matrix, impute, scale ---
    #     X = self.data[features].copy()

    #     imputer = SimpleImputer(strategy="median")
    #     X_imputed = imputer.fit_transform(X)

    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X_imputed)

    #     # --- choose k by silhouette if not provided ---
    #     if n_clusters is None:
    #         best_k = None
    #         best_score = -1.0
    #         for k in range(k_search_range[0], k_search_range[1] + 1):
    #             if k < 2:
    #                 continue
    #             km_try = KMeans(n_clusters=k, random_state=42, n_init=10)
    #             labels_try = km_try.fit_predict(X_scaled)
    #             if (
    #                 len(set(labels_try)) < 2
    #                 or len(set(labels_try)) >= X_scaled.shape[0]
    #             ):
    #                 continue
    #             try:
    #                 score = silhouette_score(X_scaled, labels_try)
    #             except Exception:
    #                 score = -1.0
    #             print(f"Silhouette for k={k}: {score:.4f}")
    #             if score > best_score:
    #                 best_score = score
    #                 best_k = k
    #         if best_k is None:
    #             best_k = 3
    #             print("Could not pick best k by silhouette; falling back to k=3")
    #         print(f"Selected k = {best_k} (silhouette {best_score:.4f})")
    #         n_clusters = best_k

    #     # --- final KMeans ---
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    #     labels = kmeans.fit_predict(X_scaled)
    #     self.data["cluster"] = labels

    #     # --- summaries ---
    #     cluster_summary = self.data.groupby("cluster")[features].agg(
    #         ["mean", "std", "count"]
    #     )
    #     print("Cluster summary (mean, std, count):")
    #     print(cluster_summary)

    #     patients_by_cluster = self.data.groupby("cluster")["patient_id"].apply(list)
    #     for c, plist in patients_by_cluster.items():
    #         print(f"\nCluster {c} ({len(plist)} patients):\n{plist}")

    #     # --- PCA visualization ---
    #     pca = PCA(n_components=3, random_state=42)
    #     X_pca = pca.fit_transform(X_scaled)

    #     fig, ax = plt.subplots(figsize=(9, 6))
    #     scatter = ax.scatter(
    #         X_pca[:, 0],
    #         X_pca[:, 1],
    #         c=labels,
    #         cmap="tab10",
    #         s=80,
    #         edgecolor="k",
    #         linewidth=0.5,
    #     )

    #     # annotate patient ids (optional — remove or reduce font size if too crowded)
    #     for i, pid in enumerate(self.data["patient_id"].astype(str).values):
    #         ax.annotate(pid, (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.9)

    #     ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    #     ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    #     ax.set_title(f"KMeans clustering (k={n_clusters}) - PCA projection")
    #     legend1 = ax.legend(*scatter.legend_elements(), title="cluster")
    #     ax.add_artist(legend1)
    #     plt.tight_layout()

    #     if plot_path:
    #         fig.savefig(plot_path, dpi=200)
    #         print(f"Saved cluster PCA plot to {plot_path}")
    #         plt.close(fig)
    #     else:
    #         plt.show()

    #     return {
    #         "kmeans": kmeans,
    #         "scaler": scaler,
    #         "imputer": imputer,
    #         "features": features,
    #         "pca": pca,
    #     }

    def kmeans_clustering(
        self,
        n_clusters: int | None = None,
        k_search_range=(2, 6),
        plot_path: str | None = None,
    ):
        """
        KMeans clustering on stressind_* features excluding pct_* and numeric-percentage slots (20/40/...).
        - n_clusters: if provided, use that. If None, pick by silhouette in k_search_range.
        - plot_path: if provided, save the PCA scatter to that path; otherwise show it.
        """

        # sanity
        if self.data is None or self.data.empty:
            raise ValueError("No patient data loaded. Run load_patient_data first.")

        # --- safe pc1 delta computation (creates stressind_*_pc1_change and rel_change) ---
        def safe_compute_delta(df, s_col, d_col, out_prefix):
            if s_col in df.columns and d_col in df.columns:
                delta = df[s_col] - df[d_col]
                rel = np.where(df[d_col].abs() > 1e-12, delta / df[d_col], np.nan)
                df[f"{out_prefix}_change"] = delta
                df[f"{out_prefix}_rel_change"] = rel
            else:
                logger.debug(
                    f"Skipping delta creation for missing columns: {s_col}, {d_col}"
                )

        # dia ost & mla
        safe_compute_delta(
            self.data,
            "stressind_dia_pc1_systole_ost",
            "stressind_dia_pc1_diastole_ost",
            "dia_pc1_ost",
        )
        safe_compute_delta(
            self.data,
            "stressind_dia_pc1_systole_mla",
            "stressind_dia_pc1_diastole_mla",
            "dia_pc1_mla",
        )

        # sys ost & mla
        safe_compute_delta(
            self.data,
            "stressind_sys_pc1_systole_ost",
            "stressind_sys_pc1_diastole_ost",
            "sys_pc1_ost",
        )
        safe_compute_delta(
            self.data,
            "stressind_sys_pc1_systole_mla",
            "stressind_sys_pc1_diastole_mla",
            "sys_pc1_mla",
        )

        # --- pick features: only stressind_* features + the computed pc1 deltas ---
        allowed_prefixes = ("stressind_",)

        exclude_substrings = (
            "_pct_",
            "_20_",
            "_40_",
            "_60_",
            "_80_",
            "_100_",
            "_120_",
            "_140_",
            "_160_",
        )

        def keep_col(col: str) -> bool:
            if not col.startswith(allowed_prefixes):
                return False
            for ex in exclude_substrings:
                if ex in col:
                    return False
            return True

        stressind_features = [col for col in self.data.columns if keep_col(col)]

        pc1_features = [
            c
            for c in (
                "dia_pc1_ost_change",
                "dia_pc1_ost_rel_change",
                "dia_pc1_mla_change",
                "dia_pc1_mla_rel_change",
                "sys_pc1_ost_change",
                "sys_pc1_ost_rel_change",
                "sys_pc1_mla_change",
                "sys_pc1_mla_rel_change",
            )
            if c in self.data.columns
        ]

        features = stressind_features + pc1_features

        if not features:
            raise ValueError(
                "No matching features found for clustering. Check column names / prefixes."
            )

        print("Using features for clustering (count={}):".format(len(features)))
        print(features)

        # --- build matrix, impute, scale ---
        X = self.data[features].copy()

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # --- choose k by silhouette if not provided ---
        if n_clusters is None:
            best_k = None
            best_score = -1.0
            for k in range(k_search_range[0], k_search_range[1] + 1):
                if k < 2:
                    continue
                km_try = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_try = km_try.fit_predict(X_scaled)
                if (
                    len(set(labels_try)) < 2
                    or len(set(labels_try)) >= X_scaled.shape[0]
                ):
                    continue
                try:
                    score = silhouette_score(X_scaled, labels_try)
                except Exception:
                    score = -1.0
                print(f"Silhouette for k={k}: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
            if best_k is None:
                best_k = 3
                print("Could not pick best k by silhouette; falling back to k=3")
            print(f"Selected k = {best_k} (silhouette {best_score:.4f})")
            n_clusters = best_k

        # --- final KMeans ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
        labels = kmeans.fit_predict(X_scaled)
        self.data["cluster"] = labels

        # --- summaries ---
        cluster_summary = self.data.groupby("cluster")[features].agg(
            ["mean", "std", "count"]
        )
        print("Cluster summary (mean, std, count):")
        print(cluster_summary)

        patients_by_cluster = self.data.groupby("cluster")["patient_id"].apply(list)
        for c, plist in patients_by_cluster.items():
            print(f"\nCluster {c} ({len(plist)} patients):\n{plist}")

        # --- PCA visualization ---
        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            cmap="tab10",
            s=80,
            edgecolor="k",
            linewidth=0.5,
        )

        for i, pid in enumerate(self.data["patient_id"].astype(str).values):
            ax.annotate(pid, (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.9)

        ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.2%} var)")
        ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.2%} var)")
        ax.set_title(f"KMeans clustering (k={n_clusters}) - PCA projection")
        legend1 = ax.legend(*scatter.legend_elements(), title="cluster")
        ax.add_artist(legend1)
        plt.tight_layout()

        if plot_path:
            fig.savefig(plot_path, dpi=200)
            print(f"Saved cluster PCA plot to {plot_path}")
            plt.close(fig)
        else:
            plt.show()

        return {
            "kmeans": kmeans,
            "scaler": scaler,
            "imputer": imputer,
            "features": features,
            "pca": pca,
        }

    def univariate_logistic_regression(
        self, target_var_cont="pdpa_mean_dobu", target_cutoff=0.8
    ):
        """
        Perform simple (univariate) logistic regression for each numeric feature.
        Predicts a binary target derived from `target_var_cont` <= `target_cutoff`.
        Saves results including coefficients, p-values (if available), and AUC scores.
        """
        from sklearn.metrics import roc_auc_score
        import statsmodels.api as sm

        df = self.data.copy()
        df["target"] = (df[target_var_cont] <= target_cutoff).astype(int)

        results = []

        numeric_features = df.select_dtypes(include=np.number).columns.drop(
            "target", errors="ignore"
        )
        logger.info(
            f"Running univariate logistic regression on {len(numeric_features)} features."
        )

        for feature in numeric_features:
            try:
                X = df[[feature]].dropna()
                y = df.loc[X.index, "target"]

                # Skip if not enough variation
                if y.nunique() != 2 or len(X) < 10:
                    logger.debug(
                        f"Skipping {feature}: not enough class variation or data."
                    )
                    continue

                # Use statsmodels for coefficient and p-value
                X_sm = sm.add_constant(X)
                model = sm.Logit(y, X_sm).fit(disp=0)
                pred_prob = model.predict(X_sm)

                auc = roc_auc_score(y, pred_prob)

                results.append(
                    {
                        "feature": feature,
                        "coef": model.params[feature],
                        "p_value": model.pvalues[feature],
                        "auc": auc,
                        "n_samples": len(y),
                    }
                )

            except Exception as e:
                logger.warning(f"Failed on {feature}: {e}")
                continue

        results_df = pd.DataFrame(results)
        results_df = results_df[
            results_df["p_value"] < 0.05
        ]  # Filter significant features
        results_df = results_df.sort_values(by="auc", ascending=False)
        output_file = os.path.join(
            self.input_dir, f"logreg_univariate_{target_var_cont}_{target_cutoff}_local.csv"
        )
        results_df.to_csv(output_file, index=False)
        logger.success(f"Univariate logistic regression results saved to {output_file}")
