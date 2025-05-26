import os
from scipy.stats import wilcoxon, shapiro, ttest_rel, ranksums, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import NotFittedError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

class GlobalStats:
    def __init__(self, global_data: pd.DataFrame, output_dir: str):
        self.global_data = global_data
        self.output_dir = output_dir
        self._create_ouput_dir()
        output_file = os.path.join(self.output_dir, "global_data.csv")
        self.global_data.to_csv(output_file, index=False)
        self.stats_df = pd.DataFrame()

    def _create_ouput_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")

    def global_stats(self):
        self.plot_global_change()
        self.pairwise_rest_stress()
        self.pairwise_pressure_cutoff(cutoff_var="iFR_mean_stress", cutoff=0.8)
        self.pairwise_pressure_cutoff(cutoff_var="pdpa_mean_stress", cutoff=0.8)
        self.univariate_logistic_regression(target_var_cont="pdpa_mean_stress", target_cutoff=0.8)
        self.univariate_logistic_regression(target_var_cont="iFR_mean_stress", target_cutoff=0.8)

    def pairwise_rest_stress(self):
        """Compare rest vs. stress metrics using appropriate statistical tests."""
        # replace 'dobu' with 'stress' in column names
        self.global_data.columns = self.global_data.columns.str.replace('dobu', 'stress', regex=False)
        # 1. Filter numeric columns only
        numeric_cols = self.global_data.select_dtypes(include=np.number).columns.tolist()

        valid_cols = [
            col for col in self.global_data.columns 
            if ("rest" in col or "stress" in col) and col in numeric_cols
        ]
        
        # 2. Melt with filtered columns
        merged_df = self.global_data.melt(
            id_vars=["patient_id"],
            value_vars=valid_cols,
            var_name="metric_phase",
            value_name="value"
        )
        
        # 3. Correctly split metric and phase
        def extract_phase(metric_phase):
            parts = metric_phase.split('_')
            for i, part in enumerate(parts):
                if part in ['rest', 'stress']:
                    before = parts[:i]
                    after = parts[i+1:]
                    metric = '_'.join(before + after) if after else '_'.join(before)
                    return (metric, part)
            return (metric_phase, None)
        
        split = merged_df["metric_phase"].apply(extract_phase)
        merged_df[["metric", "phase"]] = pd.DataFrame(split.tolist(), index=merged_df.index)
        merged_df = merged_df.dropna(subset=["phase"])  # Remove rows where phase wasn't found
        
        # 4. Convert values to numeric, drop invalid
        merged_df["value"] = pd.to_numeric(merged_df["value"], errors="coerce")
        merged_df = merged_df.dropna(subset=["value"])
        
        # 5. Pivot and analyze
        pivoted = merged_df.pivot_table(
            index=["patient_id", "metric"],
            columns="phase",
            values="value"
        ).reset_index().dropna(subset=["rest", "stress"])  # Forcefully drop rows with NaN in either phase

        results = []
        for metric in pivoted["metric"].unique():
            subset = pivoted[pivoted["metric"] == metric]
            
            # Critical fix: Require minimum 6 paired samples for tests
            if len(subset) < 6:  # Minimum for Shapiro-Wilk (3) + buffer
                logger.debug(f"Skipping {metric}: Only {len(subset)} valid pairs")
                continue
                
            # Normality check with strict NaN handling
            try:
                _, p_rest = shapiro(subset["rest"].dropna())
                _, p_stress = shapiro(subset["stress"].dropna())
            except ValueError as e:
                logger.warning(f"Shapiro failed for {metric}: {str(e)}")
                continue

            # Ensure we have enough samples AFTER normality checks
            paired = subset[["rest", "stress"]].dropna()
            if len(paired) < 6:
                logger.debug(f"Insufficient paired samples for {metric}: {len(paired)}")
                continue

            # Choose test
            if p_rest > 0.05 and p_stress > 0.05:
                stat, p = ttest_rel(paired["rest"], paired["stress"], nan_policy='omit')
                test = "t-test"
                metric_rest = np.mean(paired["rest"])
                dev_rest = np.std(paired["rest"])
                metric_stress = np.mean(paired["stress"])
                dev_stress = np.std(paired["stress"])
            else:
                stat, p = wilcoxon(paired["rest"], paired["stress"], nan_policy='omit')
                test = "Wilcoxon"
                metric_rest = np.median(paired["rest"])
                dev_rest = np.subtract(*np.percentile(paired["rest"], [75, 25]))
                metric_stress = np.median(paired["stress"])
                dev_stress = np.subtract(*np.percentile(paired["stress"], [75, 25]))
                
            results.append({
                "metric": metric,
                "metric_rest": metric_rest,
                "dev_rest": dev_rest,
                "metric_stress": metric_stress,
                "dev_stress": dev_stress,
                "statistic": stat,
                "p_value": p,
                "test": test
            })
        
        self.stats_df = pd.DataFrame(results)
        logger.success(f"Generated stats for {len(results)} metrics")
       
    def pairwise_pressure_cutoff(self, cutoff_var="iFR_mean_stress", cutoff=0.8):
        """Compare metrics between groups above vs. below/equal to cutoff."""
        # Validate cutoff variable exists
        if cutoff_var not in self.global_data.columns:
            logger.error(f"Cutoff variable '{cutoff_var}' not found")
            return

        # Split into groups
        above_mask = self.global_data[cutoff_var] > cutoff
        group_above = self.global_data[above_mask]
        group_below = self.global_data[~above_mask]
        
        logger.info(f"Group sizes - Above: {len(group_above)}, Below/Equal: {len(group_below)}")

        results = []

        metrics = self.global_data.columns.tolist()

        for metric in metrics:
            # Get data for both groups
            data_above = group_above[metric].dropna()
            data_below = group_below[metric].dropna()
            
            # Check minimum sample sizes
            if len(data_above) < 6 or len(data_below) < 6:
                logger.debug(f"Skipping {metric}: insufficient samples (Above={len(data_above)}, Below={len(data_below)})")
                continue
                
            # Normality checks
            try:
                _, p_above = shapiro(data_above)
                _, p_below = shapiro(data_below)
                normal = (p_above > 0.05) and (p_below > 0.05)
            except ValueError as e:
                logger.warning(f"Normality check failed for {metric}: {str(e)}")
                continue
                
            # Choose test
            if normal:
                stat, p = ttest_ind(data_above, data_below)
                test = "t-test"
            else:
                stat, p = ranksums(data_above, data_below)  # Wilcoxon rank-sum
                test = "Wilcoxon"
                
            # Calculate effect sizes
            mad_above = np.median(np.abs(data_above - np.median(data_above)))
            mad_below = np.median(np.abs(data_below - np.median(data_below)))
            
            results.append({
                "metric": metric,
                "above_median": np.median(data_above),
                "above_mad": mad_above,
                "below_median": np.median(data_below),
                "below_mad": mad_below,
                "statistic": stat,
                "p_value": p,
                "test": test,
                "n_above": len(data_above),
                "n_below": len(data_below)
            })
        
        # Create and save results
        self.cutoff_stats = pd.DataFrame(results)
        output_path = os.path.join(self.output_dir, f"cutoff_{cutoff_var}_{cutoff}.csv")
        self.cutoff_stats.to_csv(output_path, index=False)
        
        logger.success(f"Generated cutoff comparison stats for {len(results)} metrics")

    def plot_global_change(self, mode="pressure", phases=None):
        """
        Plot paired boxplots for pressure, lumen, or mln data,
        with Wilcoxon p-value annotations.

        mode: "pressure", "lumen", or "mln"
        phases: list of phase suffixes for pressure.
                Ignored for lumen and mln.
        """
        # 1) Determine phases
        if mode == "lumen":
            phases = ["rest", "stress"]
        elif mode == "mln":
            phases = ["rest", "stress"]
        elif phases is None:
            phases = ["rest", "dobu"]

        # 2) Validate
        if mode not in ("pressure", "lumen", "mln"):
            logger.error(f"Invalid mode: {mode}. Use 'pressure', 'lumen', or 'mln'.")
            return
        if len(phases) < 2:
            logger.warning("At least two phases are required for plotting.")
            return

        # 3) Build column names & titles
        if mode == "pressure":
            left_cols  = [f"pdpa_mean_{ph}" for ph in phases]
            right_cols = [f"iFR_mean_{ph}"  for ph in phases]
            left_title, right_title = "PDPA Mean", "iFR Mean"
            right_hline, hline_label = 0.8, "iFR = 0.8"
        elif mode == "lumen":
            left_cols  = [f"ostial_a_{ph}" for ph in phases]
            right_cols = [f"mla_{ph}"      for ph in phases]
            left_title, right_title = "Ostial Area", "MLA Area"
            right_hline, hline_label = None, None
        else:  # mode == "mln"
            left_cols  = [f"ostial_{ph}_mln" for ph in phases]
            right_cols = [f"mla_{ph}_mln"    for ph in phases]
            left_title, right_title = "Ostial Area (MLN)", "MLA Area (MLN)"
            right_hline, hline_label = None, None

        # 4) Check columns
        missing = [c for c in left_cols + right_cols if c not in self.global_data.columns]
        if missing:
            logger.error(f"Missing columns for mode '{mode}': {missing}")
            return

        # 5) Subset & drop NaNs
        df = self.global_data[left_cols + right_cols].dropna()
        logger.info(f"Using {len(df)} rows after dropping NaNs for mode='{mode}'.")

        # 6) Create plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        for ax, cols, title, idx in zip(
            axes,
            [left_cols, right_cols],
            [left_title, right_title],
            [0, 1]
        ):
            # Boxplot + jittered lines
            df[cols].boxplot(ax=ax)
            for _, row in df.iterrows():
                ax.plot(range(1, len(phases)+1), row[cols], color="gray", alpha=0.5)
            ax.set_title(title)
            ax.set_xticklabels([p.capitalize() for p in phases])
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.6)

            # Optional horizontal line on second panel
            if idx == 1 and right_hline is not None:
                ax.axhline(right_hline, color="red", linestyle="--", label=hline_label)
                ax.legend()

            # Wilcoxon test
            try:
                stat, p = wilcoxon(df[cols[0]], df[cols[1]])
                y = df[cols].max().max() * 1.05
                ax.annotate(f"p = {p:.3f}", xy=(1.5, y), ha="center")
            except ValueError as e:
                logger.warning(f"Cannot run Wilcoxon on {cols}: {e}")

        plt.tight_layout()
        # 7) Save
        fname = f"global_{mode}_{'_'.join(phases)}.png"
        out = os.path.join(self.output_dir, fname)
        plt.savefig(out)
        logger.info(f"Saved plot to: {out}")

    def univariate_logistic_regression(self, target_var_cont="pdpa_mean_stress", target_cutoff=0.8):
        """
        Perform simple (univariate) logistic regression for each numeric feature.
        Predicts a binary target derived from `target_var_cont` <= `target_cutoff`.
        Saves results including coefficients, p-values (if available), and AUC scores.
        """
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        import statsmodels.api as sm

        df = self.global_data.copy()
        df["target"] = (df[target_var_cont] <= target_cutoff).astype(int)

        results = []

        numeric_features = df.select_dtypes(include=np.number).columns.drop("target", errors="ignore")
        logger.info(f"Running univariate logistic regression on {len(numeric_features)} features.")

        for feature in numeric_features:
            try:
                X = df[[feature]].dropna()
                y = df.loc[X.index, "target"]

                # Skip if not enough variation
                if y.nunique() != 2 or len(X) < 10:
                    logger.debug(f"Skipping {feature}: not enough class variation or data.")
                    continue

                # Use statsmodels for coefficient and p-value
                X_sm = sm.add_constant(X)
                model = sm.Logit(y, X_sm).fit(disp=0)
                pred_prob = model.predict(X_sm)

                auc = roc_auc_score(y, pred_prob)

                results.append({
                    "feature": feature,
                    "coef": model.params[feature],
                    "p_value": model.pvalues[feature],
                    "auc": auc,
                    "n_samples": len(y)
                })

            except Exception as e:
                logger.warning(f"Failed on {feature}: {e}")
                continue

        results_df = pd.DataFrame(results)
        results_df = results_df[results_df["p_value"] < 0.05]  # Filter significant features
        results_df = results_df.sort_values(by="auc", ascending=False)
        output_file = os.path.join(self.output_dir, f"logreg_univariate_{target_var_cont}_{target_cutoff}.csv")
        results_df.to_csv(output_file, index=False)
        logger.success(f"Univariate logistic regression results saved to {output_file}")