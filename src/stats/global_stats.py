import os

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

    def _create_ouput_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Output directory already exists: {self.output_dir}")
        
    def plot_global_change(self, mode="pressure", phases=None):
        """
        Plot paired boxplots for either:
          - pressure: pdpa_mean_*, iFR_mean_* 
          - lumen: ostial_a_*, mla_*

        Parameters:
          - mode: "pressure" or "lumen"
          - phases: list of suffixes to include. If None:
              * pressure → ["rest", "dobu"]
              * lumen    → ["rest", "stress"]
            Overrides for pressure only; lumen always uses rest & stress.
        """
        # Determine phases based on mode
        if mode == "lumen":
            phases = ["rest", "stress"]
        elif phases is None:
            phases = ["rest", "dobu"]

        # Validate mode
        if mode not in ("pressure", "lumen"):
            logger.error(f"Invalid mode: {mode}. Use 'pressure' or 'lumen'.")
            return

        # Validate enough phases
        if len(phases) < 2:
            logger.warning("At least two phases are required for plotting.")
            return

        # Configure prefixes and titles
        if mode == "pressure":
            left_base, right_base = "pdpa_mean_", "iFR_mean_"
            left_title, right_title = "PDPA Mean", "iFR Mean"
            right_hline, right_hline_label = 0.8, "iFR = 0.8"
        else:  # lumen
            left_base, right_base = "ostial_a_", "mla_"
            left_title, right_title = "Ostial Area", "MLA Area"
            right_hline, right_hline_label = None, None

        # Build column lists and verify
        left_cols  = [f"{left_base}{ph}" for ph in phases]
        right_cols = [f"{right_base}{ph}" for ph in phases]
        missing = [c for c in left_cols + right_cols if c not in self.global_data.columns]
        if missing:
            logger.error(f"Missing columns for mode '{mode}': {missing}")
            return

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

        # Left subplot
        ax = axes[0]
        self.global_data[left_cols].boxplot(ax=ax)
        ax.set_title(left_title)
        ax.set_ylabel("Value")
        ax.set_xticklabels([ph.capitalize() for ph in phases])
        ax.grid(True, linestyle="--", alpha=0.6)
        for _, row in self.global_data.iterrows():
            ax.plot(range(1, len(phases)+1), row[left_cols], color="gray", alpha=0.7, linewidth=1)

        # Right subplot
        ax = axes[1]
        self.global_data[right_cols].boxplot(ax=ax)
        ax.set_title(right_title)
        ax.set_ylabel("Value")
        ax.set_xticklabels([ph.capitalize() for ph in phases])
        if right_hline is not None:
            ax.axhline(right_hline, color="red", linestyle="--", label=right_hline_label)
            ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        for _, row in self.global_data.iterrows():
            ax.plot(range(1, len(phases)+1), row[right_cols], color="gray", alpha=0.7, linewidth=1)

        plt.tight_layout()
        filename = f"global_{mode}_{'_'.join(phases)}.png"
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path)
        # plt.show()
        logger.info(f"Saved plot to: {out_path}")
