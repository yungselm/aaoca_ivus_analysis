from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def _create_summary_displacement(path: Path) -> dict:
    """
    For each subdirectory under `path`, looks for these CSVs:
      - rest_sys_dia_displacement_map.csv
      - stress_sys_dia_displacement_map.csv
      - dia_dia_displacement_map.csv
      - sys_sys_displacement_map.csv

    If found, loads each into a DataFrame, pads/trims so each has columns
    bin_0 … bin_9 (filling missing ones with NaN), and collects them.
    In the end, returns a dict of 4 DataFrames (one per phase), where each
    output DataFrame has the same shape as an individual CSV and each cell is
    the mean across that cell’s values over all folders—ignoring NaNs.
    """
    # We want exactly bin_0 ... bin_9
    required_bins = [f"bin_{i}" for i in range(10)]

    # Lists of standardized DataFrames for each phase
    rest_list = []
    stress_list = []
    dia_dia_list = []
    sys_sys_list = []

    for subdir in os.listdir(path):
        subpath = os.path.join(path, subdir)
        if not os.path.isdir(subpath):
            continue
        print(subpath)
        filenames = {
            "rest": "rest_sys_dia_displacement_map.csv",
            "stress": "stress_sys_dia_displacement_map.csv",
            "dia_dia": "dia_dia_displacement_map.csv",
            "sys_sys": "sys_sys_displacement_map.csv",
        }

        try:
            df_rest = pd.read_csv(os.path.join(subpath, filenames["rest"]))
            df_stress = pd.read_csv(os.path.join(subpath, filenames["stress"]))
            df_dia = pd.read_csv(os.path.join(subpath, filenames["dia_dia"]))
            df_sys = pd.read_csv(os.path.join(subpath, filenames["sys_sys"]))
        except FileNotFoundError:
            # Skip this folder if any required CSV is missing
            continue

        # Helper: ensure this DataFrame has exactly required_bins in order,
        # filling missing columns with NaN.
        def _standardize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # Add any missing bin_i column as NaN
            for col in required_bins:
                if col not in df.columns:
                    df[col] = np.nan
            # Select only bin_0..bin_9 in that exact order
            return df[required_bins]

        rest_list.append(_standardize(df_rest))
        stress_list.append(_standardize(df_stress))
        dia_dia_list.append(_standardize(df_dia))
        sys_sys_list.append(_standardize(df_sys))

    def _average_cellwise_ignore_nan(df_list: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Given a list of DataFrames (all the same shape, same index & columns),
        stack them vertically, then group by row‐index and take mean (skipna=True).
        Returns a DataFrame of shape (n_rows, 11) with each cell = mean over
        that cell across all DataFrames, ignoring NaNs.
        """
        if not df_list:
            # Return an empty DataFrame with the correct columns but zero rows
            return pd.DataFrame(columns=required_bins)

        # Concatenate along axis=0 (rows). Each DataFrame shares the same row indices 0..(R-1).
        concatenated = pd.concat(df_list, axis=0, ignore_index=False)

        # Group by the original row index (0..R-1), then take mean over each column.
        # By default, .mean() skips NaN values.
        averaged = concatenated.groupby(concatenated.index).mean()

        # Ensure the result has exactly required_bins (just in case)
        return averaged[required_bins]

    return {
        "rest_sys_dia": _average_cellwise_ignore_nan(rest_list),
        "stress_sys_dia": _average_cellwise_ignore_nan(stress_list),
        "dia_dia": _average_cellwise_ignore_nan(dia_dia_list),
        "sys_sys": _average_cellwise_ignore_nan(sys_sys_list),
    }


def plot_all_heatmaps(displacement_dict: dict, cmap="viridis") -> None:
    """
    Given a dictionary where each key maps to a DataFrame of shape (n_rows, 11)
    with columns ["bin_0", ..., "bin_10"], generate a Seaborn heatmap for each entry.

    Parameters
    ----------
    displacement_dict : dict[str, pandas.DataFrame]
        Keys are phase names (e.g., "rest_sys_dia", "stress_sys_dia", etc.)
        Values are DataFrames where rows correspond to point bins and
        columns correspond to contour bins ("bin_0" ... "bin_10").
    cmap : str, optional
        Matplotlib colormap name for the heatmaps. Default is "viridis".
    """
    for phase, df in displacement_dict.items():
        if df.empty:
            # Skip plotting if the DataFrame is empty
            continue

        plt.figure(figsize=(8, 6))
        # Use seaborn’s heatmap; df’s index is point‐bin index, columns are contour‐bin
        sns.heatmap(
            df,
            cmap=cmap,
            cbar_kws={"label": "Mean Displacement"},
            xticklabels=df.columns,
            yticklabels=df.index,
        )
        # Custom x labels: 0=0-10°, 1=10-20°, ..., 10=100-110°
        y_labels = [f"{i*10}-{(i+1)*10}°" for i in range(36)]
        # Custom y labels for bins
        x_labels = [
            "IM 0-20%",
            "IM 20-40%",
            "IM 40-60%",
            "IM 60-80%",
            "IM 80-100%",
            "EM 100-120%",
            "EM 120-140%",
            "EM 140-160%",
            "EM 160-180%",
            "EM 180-200%",
        ]
        plt.title(f"{phase.replace('_', ' ').title()} Heatmap")
        plt.xlabel("Vessel position intramural (IM) / extramural (EM)")
        plt.ylabel("Circumferential position (°)")
        plt.xticks(ticks=np.arange(len(df.columns)) + 0.5, labels=x_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(df.index)) + 0.5, labels=y_labels, rotation=0)
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.show()


def load_measurement_data(path: str) -> dict[str, pd.DataFrame]:
    """
    Walks through each subdirectory of `path`. In each subdirectory we expect to find:
      - rest_measurement_map.csv
      - stress_measurement_map.csv
      - dia_dia_measurement_map.csv
      - sys_sys_measurement_map.csv

    Each CSV is read with index_col=0 so that “bin_0”… become the index. We then:
      • Reset index → a column “bin”
      • Drop any bins beyond bin_9
      • Pad missing bins in bin_0…bin_9 with NaNs
      • Concatenate into four master DataFrames (df_rest, df_stress, df_dia, df_sys)
        whose columns and dtypes are fixed up front (all floats for measurement columns).

    Returns a dict with keys "rest", "stress", "dia_dia", "sys_sys".
    """

    # Define the exact required bins and their string names:
    required_bins = [f"bin_{i}" for i in range(10)]

    # 1) Initialize each master DataFrame with the correct columns *and* dtypes from the start:
    df_rest = pd.DataFrame(
        {
            "bin": pd.Series(dtype="object"),
            "lumen_area_dia": pd.Series(dtype="float64"),
            "lumen_area_sys": pd.Series(dtype="float64"),
            "min_dist_dia": pd.Series(dtype="float64"),
            "min_dist_sys": pd.Series(dtype="float64"),
            "elliptic_ratio_dia": pd.Series(dtype="float64"),
            "elliptic_ratio_sys": pd.Series(dtype="float64"),
        }
    )
    df_stress = pd.DataFrame(
        {
            "bin": pd.Series(dtype="object"),
            "lumen_area_dia": pd.Series(dtype="float64"),
            "lumen_area_sys": pd.Series(dtype="float64"),
            "min_dist_dia": pd.Series(dtype="float64"),
            "min_dist_sys": pd.Series(dtype="float64"),
            "elliptic_ratio_dia": pd.Series(dtype="float64"),
            "elliptic_ratio_sys": pd.Series(dtype="float64"),
        }
    )
    df_dia = pd.DataFrame(
        {
            "bin": pd.Series(dtype="object"),
            "lumen_area_dia_rest": pd.Series(dtype="float64"),
            "lumen_area_dia_stress": pd.Series(dtype="float64"),
            "min_dist_dia_rest": pd.Series(dtype="float64"),
            "min_dist_dia_stress": pd.Series(dtype="float64"),
            "elliptic_ratio_dia_rest": pd.Series(dtype="float64"),
            "elliptic_ratio_dia_stress": pd.Series(dtype="float64"),
        }
    )
    df_sys = pd.DataFrame(
        {
            "bin": pd.Series(dtype="object"),
            "lumen_area_sys_rest": pd.Series(dtype="float64"),
            "lumen_area_sys_stress": pd.Series(dtype="float64"),
            "min_dist_sys_rest": pd.Series(dtype="float64"),
            "min_dist_sys_stress": pd.Series(dtype="float64"),
            "elliptic_ratio_sys_rest": pd.Series(dtype="float64"),
            "elliptic_ratio_sys_stress": pd.Series(dtype="float64"),
        }
    )

    filenames = {
        "rest": "rest_measurement_map.csv",
        "stress": "stress_measurement_map.csv",
        "dia_dia": "dia_dia_measurement_map.csv",
        "sys_sys": "sys_sys_measurement_map.csv",
    }

    def _check_len_and_add(df: pd.DataFrame, all_bins: list[str]) -> pd.DataFrame:
        """
        Expects a DataFrame `df` with a column “bin” containing some subset of all_bins.
        1) Drop any rows whose bin is not in all_bins.
        2) If any bin_i in all_bins is missing, append one row with that bin and NaNs.
           Returns a DataFrame with exactly one row per bin in all_bins (order not guaranteed).
        """
        # 1) Filter out any “bin” not in all_bins (i.e. keep only bin_0..bin_9):
        df = df[df["bin"].isin(all_bins)].copy()

        # 2) Identify which required_bins are missing
        existing_bins = set(df["bin"].values)
        missing_bins = [b for b in all_bins if b not in existing_bins]
        if missing_bins:
            # Build a new DataFrame of missing rows: one row per missing bin, all other cols = NaN
            rows_to_add = []
            for b in missing_bins:
                nan_row = {"bin": b}
                for col in df.columns:
                    if col != "bin":
                        nan_row[col] = np.nan
                rows_to_add.append(nan_row)
            df_extra = pd.DataFrame(rows_to_add, columns=df.columns)
            # Concatenate (pd.concat avoids the .append() deprecation)
            df = pd.concat([df, df_extra], ignore_index=True)

        return df

    # Now walk each subdirectory under `path`:
    for subdir in os.listdir(path):
        subpath = os.path.join(path, subdir)
        if not os.path.isdir(subpath):
            continue

        try:
            # Read each CSV, using index_col=0 so that “bin_0”… becomes the index:
            df_rest_temp = pd.read_csv(
                os.path.join(subpath, filenames["rest"]), index_col=0
            )
            df_stress_temp = pd.read_csv(
                os.path.join(subpath, filenames["stress"]), index_col=0
            )
            df_dia_temp = pd.read_csv(
                os.path.join(subpath, filenames["dia_dia"]), index_col=0
            )
            df_sys_temp = pd.read_csv(
                os.path.join(subpath, filenames["sys_sys"]), index_col=0
            )
        except FileNotFoundError:
            # If any of the four CSVs is missing, skip this patient folder
            continue

        # Reset index so the old index (“bin_0”, “bin_1”, …) becomes a column named “bin”
        df_rest_temp = df_rest_temp.reset_index().rename(columns={"index": "bin"})
        df_stress_temp = df_stress_temp.reset_index().rename(columns={"index": "bin"})
        df_dia_temp = df_dia_temp.reset_index().rename(columns={"index": "bin"})
        df_sys_temp = df_sys_temp.reset_index().rename(columns={"index": "bin"})

        # 3) Filter out any bins > bin_9, then pad missing bins so each has exactly 10 rows:
        df_rest_temp = _check_len_and_add(df_rest_temp, required_bins)
        df_stress_temp = _check_len_and_add(df_stress_temp, required_bins)
        df_dia_temp = _check_len_and_add(df_dia_temp, required_bins)
        df_sys_temp = _check_len_and_add(df_sys_temp, required_bins)

        # 4) Concatenate onto the master DataFrames. Because we initialized
        #    each master with the correct column names & dtypes, no “all-NaN columns” issue arises.
        df_rest = pd.concat([df_rest, df_rest_temp], ignore_index=True)
        df_stress = pd.concat([df_stress, df_stress_temp], ignore_index=True)
        df_dia = pd.concat([df_dia, df_dia_temp], ignore_index=True)
        df_sys = pd.concat([df_sys, df_sys_temp], ignore_index=True)

    # 5) Finally, re‐order columns in each master DataFrame to match exactly what you expect:
    df_rest = df_rest[
        [
            "bin",
            "lumen_area_dia",
            "lumen_area_sys",
            "min_dist_dia",
            "min_dist_sys",
            "elliptic_ratio_dia",
            "elliptic_ratio_sys",
        ]
    ]
    df_stress = df_stress[
        [
            "bin",
            "lumen_area_dia",
            "lumen_area_sys",
            "min_dist_dia",
            "min_dist_sys",
            "elliptic_ratio_dia",
            "elliptic_ratio_sys",
        ]
    ]
    df_dia = df_dia[
        [
            "bin",
            "lumen_area_dia_rest",
            "lumen_area_dia_stress",
            "min_dist_dia_rest",
            "min_dist_dia_stress",
            "elliptic_ratio_dia_rest",
            "elliptic_ratio_dia_stress",
        ]
    ]
    df_sys = df_sys[
        [
            "bin",
            "lumen_area_sys_rest",
            "lumen_area_sys_stress",
            "min_dist_sys_rest",
            "min_dist_sys_stress",
            "elliptic_ratio_sys_rest",
            "elliptic_ratio_sys_stress",
        ]
    ]

    return {"rest": df_rest, "stress": df_stress, "dia_dia": df_dia, "sys_sys": df_sys}


def plot_heatmap_and_paired_violins_or_box(
    phase: str,
    disp_df: pd.DataFrame,
    meas_df: pd.DataFrame,
    cmap_heat: str = "viridis",
    box_plot: bool = False,
    disp_max: float = 0.0,
    disp_min: float = 0.0,
) -> None:
    """
    Draws a figure with two stacked subplots:
      1) Top: a heatmap of mean displacement (disp_df), with custom IM/EM and ° labels.
      2) Bottom: for each of the 10 contour-bins, two side-by-side violins or boxes:
           • diastolic (blue gradient) at x = i - 0.2
           • systolic  (red  gradient) at x = i + 0.2

    Parameters
    ----------
    phase : str
        e.g. "rest", "stress", "dia_dia", "sys_sys"
    disp_df : pd.DataFrame
        Shape (36, 10), columns = ["bin_0", …, "bin_9"], index = 0…35.
    meas_df : pd.DataFrame
        Long-form with columns:
            ["bin", "lumen_area_dia", "lumen_area_sys"].
        Each row is one patientxbin. "bin" must be "bin_0"… "bin_9".
    cmap_heat : str
        Colormap for heatmap (default "viridis").
    box_plot : bool
        If True, draw boxplots; else draw violinplots.
    """

    # 1) Derive dimensions
    n_radial_bins = disp_df.shape[0]  # should be 36
    n_circumferential_bins = disp_df.shape[1]  # should be 10

    # 2) Prepare IM/EM x_labels and ° y_labels for the heatmap
    x_labels = [
        "IM 0-20%",
        "IM 20-40%",
        "IM 40-60%",
        "IM 60-80%",
        "IM 80-100%",
        "EM 100-120%",
        "EM 120-140%",
        "EM 140-160%",
        "EM 160-180%",
        "EM 180-200%",
    ]
    # Only label 0°, 90°, 180°, 270°, 360°
    y_labels = []
    for i in range(n_radial_bins):
        deg = i * 10
        if deg in [0, 90, 180, 270, 360]:
            y_labels.append(f"{deg}°")
        else:
            y_labels.append("")

    dia_color = "#1f77b4"
    sys_color = "#d62728"

    # 4) Set up figure with two rows
    fig = plt.figure(constrained_layout=True, figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    # ===== TOP: HEATMAP =====
    ax_heat = fig.add_subplot(gs[0, 0])
    if disp_max == 0 and disp_min == 0:
        sns.heatmap(
            disp_df,
            ax=ax_heat,
            cmap=cmap_heat,
            cbar_kws={"label": "Mean displacement"},
            xticklabels=False,
            yticklabels=False,
        )
    else:
        sns.heatmap(
            disp_df,
            ax=ax_heat,
            cmap=cmap_heat,
            cbar_kws={"label": "Mean displacement"},
            xticklabels=False,
            yticklabels=False,
            vmin=disp_min,
            vmax=disp_max,
        )
    ax_heat.set_title(f"{phase.replace('_',' ').title()} ― Mean Displacement")
    ax_heat.set_xlabel("Vessel position intramural (IM) / extramural (EM)")
    ax_heat.set_ylabel("Circumferential position (°)")

    # Set custom x‐ and y‐ticks at center‐of‐cell
    ax_heat.set_xticks(np.arange(n_circumferential_bins) + 0.5)
    ax_heat.set_xticklabels(x_labels, rotation=45, ha="center")
    ax_heat.set_yticks(np.arange(n_radial_bins) + 0.5)
    ax_heat.set_yticklabels(y_labels, rotation=0)
    ax_heat.set_xlim(-0.5, n_radial_bins - 0.5)
    ax_heat.invert_xaxis()

    # Draw a horizontal black line at y=0, from the middle (x=5) to the right edge (accounting for inverted x-axis)
    ax_heat.hlines(y=0.5, xmin=1.5-n_circumferential_bins, xmax=5, color='violet', linestyle='--')
    ax_heat.hlines(y=n_radial_bins // 2 + 0.5, xmin=1-n_circumferential_bins, xmax=5, color='orange', linestyle='--')
    ax_heat.hlines(y=n_radial_bins - 0.5, xmin=1-n_circumferential_bins, xmax=5, color='violet', linestyle='--')
    ax_heat.hlines(y=n_radial_bins // 4 + 0.5, xmin=1-n_circumferential_bins, xmax=5, color='red', linestyle='--')
    ax_heat.hlines(y=(n_radial_bins - n_radial_bins // 4) + 0.5, xmin=1-n_circumferential_bins, xmax=5, color='blue', linestyle='--')

    # Draw a vertical line in the middle (between IM and EM)
    ax_heat.axvline(5, color='black', linewidth=2)

    # ===== BOTTOM: VIOLINS or BOXES =====
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_heat)

    # Convert meas_df → long‐form if needed (here we assume it already has the two columns)
    measure = meas_df.copy()
    measure["bin_idx"] = measure["bin"].str.replace("bin_", "").astype(int)

    # Plot either violin or box for each bin
    for i in range(n_circumferential_bins):
        sub = measure[measure["bin_idx"] == i]
        if sub.empty:
            continue
        
        if phase=='rest' or phase=='stress':
            dia_vals = sub["lumen_area_dia"].dropna().values
            sys_vals = sub["lumen_area_sys"].dropna().values
        elif phase=='dia_dia':
            dia_vals = sub["lumen_area_dia_rest"].dropna().values
            sys_vals = sub["lumen_area_dia_stress"].dropna().values
        else:
            dia_vals = sub["lumen_area_sys_rest"].dropna().values
            sys_vals = sub["lumen_area_sys_stress"].dropna().values
        
        pos_d = i + 0.2
        pos_s = i - 0.2

        if box_plot:
            # ========== BOXES ==========
            if dia_vals.size > 0:
                bp = ax_v.boxplot(
                    dia_vals,
                    positions=[pos_d],
                    widths=0.3,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(dia_color)

            if sys_vals.size > 0:
                bp = ax_v.boxplot(
                    sys_vals,
                    positions=[pos_s],
                    widths=0.3,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(sys_color)

        else:
            # ========== VIOLINS ==========
            if dia_vals.size > 0:
                vp = ax_v.violinplot(
                    dia_vals, positions=[pos_d], widths=0.3, showextrema=False
                )
                vp["bodies"][0].set_facecolor(dia_color)
                vp["bodies"][0].set_edgecolor("black")
                vp["bodies"][0].set_alpha(0.8)

            if sys_vals.size > 0:
                vp = ax_v.violinplot(
                    sys_vals, positions=[pos_s], widths=0.3, showextrema=False
                )
                vp["bodies"][0].set_facecolor(sys_color)
                vp["bodies"][0].set_edgecolor("black")
                vp["bodies"][0].set_alpha(0.8)

    # Tidy up bottom axis
    ax_v.set_title(f"{phase.replace('_',' ').title()} ― Lumen Area per Contour Bin")
    ax_v.set_xlabel("Vessel position intramural (IM) / extramural (EM)")
    ax_v.set_ylabel("Lumen Area")
    ax_v.set_xticks(np.arange(n_circumferential_bins))
    ax_v.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_v.set_xlim(-0.5, n_circumferential_bins - 0.5)
    ax_v.invert_xaxis()

    # Legend: one blue patch (dia) and one red patch (sys) at midpoint bin
    if phase == 'rest' or phase =='stress':
        dia_patch = Patch(facecolor=dia_color, label="Lumen Area Diastolic")
        sys_patch = Patch(facecolor=sys_color, label="Lumen Area Systolic")
    else:
        dia_patch = Patch(facecolor=dia_color, label="Lumen Area Rest")
        sys_patch = Patch(facecolor=sys_color, label="Lumen Area Stress")        
    ax_v.legend(handles=[dia_patch, sys_patch], loc="upper right")

    plt.show()


if __name__ == "__main__":
    disp_dict = _create_summary_displacement("data/output/patient_stats")
    meas_dict = load_measurement_data("data/output/patient_stats")
    # find max value and min value from disp_dict
    disp_min = float("inf")
    disp_max = float("-inf")
    for df in disp_dict.values():
        if not df.empty:
            disp_min = min(disp_min, np.nanmin(df.values))
            disp_max = max(disp_max, np.nanmax(df.values))
    print(f"Displacement min: {disp_min}, max: {disp_max}")
    # plot_all_heatmaps(disp_dict, cmap="coolwarm")
    plot_heatmap_and_paired_violins_or_box(
        phase="rest",
        disp_df=disp_dict["rest_sys_dia"],
        meas_df=meas_dict["rest"],
        cmap_heat="coolwarm",
        box_plot=True,
        disp_max=disp_max,
        disp_min=disp_min,
    )
    plot_heatmap_and_paired_violins_or_box(
        phase='stress',
        disp_df=disp_dict["stress_sys_dia"],
        meas_df=meas_dict["stress"],
        cmap_heat="coolwarm",
        box_plot=True,
        disp_max=disp_max,
        disp_min=disp_min,
    )
    plot_heatmap_and_paired_violins_or_box(
        phase="dia_dia", 
        disp_df=disp_dict["dia_dia"], 
        meas_df=meas_dict["dia_dia"],
        cmap_heat="coolwarm",
        box_plot=True,
        disp_max=disp_max,
        disp_min=disp_min,
    )
    plot_heatmap_and_paired_violins_or_box(
        phase='sys_sys',
        disp_df=disp_dict["sys_sys"], 
        meas_df=meas_dict["sys_sys"],
        cmap_heat="coolwarm",
        box_plot=True,
        disp_max=disp_max,
        disp_min=disp_min,
    )
