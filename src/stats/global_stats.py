import os
from scipy.stats import wilcoxon, shapiro, ttest_rel

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

    def pairwise_rest_stress(self):
        """Compare rest vs. stress metrics using appropriate statistical tests."""
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
        print(self.stats_df)
       
    def pairwise_pressure_cutoff(self):
        pass

    # def plot_global_change(self, mode="pressure", phases=None):
    #     """
    #     Plot paired boxplots for either:
    #       - pressure: pdpa_mean_*, iFR_mean_* 
    #       - lumen: ostial_a_*, mla_*
    #         mln: ostial_*_mln, mla_*_mln

    #     Parameters:
    #       - mode: "pressure" or "lumen"
    #       - phases: list of suffixes to include. If None:
    #           * pressure → ["rest", "dobu"]
    #           * lumen    → ["rest", "stress"]
    #         Overrides for pressure only; lumen always uses rest & stress.
    #     """
    #     # Determine phases based on mode
    #     if mode == "lumen":
    #         phases = ["rest", "stress"]
    #     elif phases is None:
    #         phases = ["rest", "dobu"]

    #     # Validate mode
    #     if mode not in ("pressure", "lumen"):
    #         logger.error(f"Invalid mode: {mode}. Use 'pressure' or 'lumen'.")
    #         return

    #     # Validate enough phases
    #     if len(phases) < 2:
    #         logger.warning("At least two phases are required for plotting.")
    #         return

    #     # Configure prefixes and titles
    #     if mode == "pressure":
    #         left_base, right_base = "pdpa_mean_", "iFR_mean_"
    #         left_title, right_title = "PDPA Mean", "iFR Mean"
    #         right_hline, right_hline_label = 0.8, "iFR = 0.8"
    #     elif mode == "lumen":  # lumen
    #         left_base, right_base = "ostial_a_", "mla_"
    #         left_title, right_title = "Ostial Area", "MLA Area"
    #         right_hline, right_hline_label = None, None
    #     else: # mln
    #         left_base, right_base = "ostial_", "mla_"
    #         left_title, right_title = "Ostial Area (MLN)", "MLA Area (MLN)"
    #         right_hline, right_hline_label = None, None

    #     # Build column lists and verify
    #     left_cols  = [f"{left_base}{ph}" for ph in phases]
    #     right_cols = [f"{right_base}{ph}" for ph in phases]
    #     missing = [c for c in left_cols + right_cols if c not in self.global_data.columns]
    #     if missing:
    #         logger.error(f"Missing columns for mode '{mode}': {missing}")
    #         return

    #     # Create plots
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    #     # Left subplot
    #     ax = axes[0]
    #     self.global_data[left_cols].boxplot(ax=ax)
    #     ax.set_title(left_title)
    #     ax.set_ylabel("Value")
    #     ax.set_xticklabels([ph.capitalize() for ph in phases])
    #     if right_hline is not None:
    #         ax.axhline(right_hline, color="red", linestyle="--", label="PdPa = 0.8")
    #         ax.legend()
    #     ax.grid(True, linestyle="--", alpha=0.6)
    #     for _, row in self.global_data.iterrows():
    #         ax.plot(range(1, len(phases)+1), row[left_cols], color="gray", alpha=0.7, linewidth=1)

    #     # Right subplot
    #     ax = axes[1]
    #     self.global_data[right_cols].boxplot(ax=ax)
    #     ax.set_title(right_title)
    #     ax.set_ylabel("Value")
    #     ax.set_xticklabels([ph.capitalize() for ph in phases])
    #     if right_hline is not None:
    #         ax.axhline(right_hline, color="red", linestyle="--", label=right_hline_label)
    #         ax.legend()
    #     ax.grid(True, linestyle="--", alpha=0.6)
    #     for _, row in self.global_data.iterrows():
    #         ax.plot(range(1, len(phases)+1), row[right_cols], color="gray", alpha=0.7, linewidth=1)

    #     plt.tight_layout()
    #     filename = f"global_{mode}_{'_'.join(phases)}.png"
    #     out_path = os.path.join(self.output_dir, filename)
    #     plt.savefig(out_path)
    #     # plt.show()
    #     logger.info(f"Saved plot to: {out_path}")

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
