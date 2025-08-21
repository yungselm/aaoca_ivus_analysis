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


class PatientStats:
    def __init__(self, input_dir: str, output_path: str):
        self.input_dir = input_dir
        self.output_path = output_path
        self.data = self.load_patient_data()

    def run(self):
        self.kmeans_clustering(n_clusters=3)

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

    def process_local_patient_data(self) -> pd.DataFrame:
        pass

    def exploratory_data_analysis(self):
        pass

    def kmeans_clustering(
        self,
        n_clusters: int | None = None,
        k_search_range=(2, 6),
        plot_path: str | None = None,
    ):
        """
        KMeans clustering on pulsatile_rest features excluding pct_* and numeric-percentage slots (20/40/...).
        - n_clusters: if provided, use that. If None, pick by silhouette in k_search_range.
        - plot_path: if provided, save the PCA scatter to that path; otherwise show it.
        """

        # sanity
        if self.data is None or self.data.empty:
            raise ValueError("No patient data loaded. Run load_patient_data first.")

        # --- safe pc1 delta computation (creates pc1_ost_change, pc1_ost_rel_change, pc1_mla_change, pc1_mla_rel_change) ---
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

        safe_compute_delta(
            self.data,
            "pulsatile_rest_pc1_systole_ost",
            "pulsatile_rest_pc1_diastole_ost",
            "pc1_ost",
        )
        safe_compute_delta(
            self.data,
            "pulsatile_rest_pc1_systole_mla",
            "pulsatile_rest_pc1_diastole_mla",
            "pc1_mla",
        )

        # --- pick features: only pulsatile_rest_ features + the two pressure/time change cols, but exclude pct_* and _20_, _40_, etc. ---
        allowed_prefixes = ("pulsatile_rest_",)

        # substrings to exclude (the pct groups and numeric percent markers)
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
            # starts with allowed prefixes AND does not contain any exclude substring
            if not col.startswith(allowed_prefixes):
                return False
            for ex in exclude_substrings:
                if ex in col:
                    return False
            return True

        rest_features = [col for col in self.data.columns if keep_col(col)]

        # include pc1 features if they were created
        pc1_features = [
            c
            for c in (
                "pc1_ost_change",
                "pc1_ost_rel_change",
                "pc1_mla_change",
                "pc1_mla_rel_change",
            )
            if c in self.data.columns
        ]

        features = rest_features + pc1_features

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

        # annotate patient ids (optional — remove or reduce font size if too crowded)
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
