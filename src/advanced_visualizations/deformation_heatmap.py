from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# def load_normalized_displacements(
#     filename: str,
#     points_per_contour: int
# ) -> np.ndarray:
#     """
#     Reads your displacement texture PNG and returns a 2D array of
#     shape (num_contours, points_per_contour) with normalized
#     displacements in [0.0, 1.0].

#     Assumes:
#       - width  == points_per_contour
#       - height == num_contours
#       - R channel = normalized_disp*255
#       - Y axis was flipped in Rust:
#           y_written = (height - 1) - contour_index
#     """
#     # load and force RGB
#     img = Image.open(filename).convert("RGB")
#     arr = np.asarray(img, dtype=np.uint8)       # shape = (H, W, 3)
#     height, width, _ = arr.shape

#     if width != points_per_contour:
#         raise ValueError(
#             f"Expected width={points_per_contour}, got {width}"
#         )

#     # Extract red channel and normalize to [0,1]
#     red = arr[:, :, 0].astype(np.float64) / 255.0

#     # Undo the Y-flip so that contour 0 is row 0, etc.
#     num_contours = height
#     displacements = np.empty((num_contours, width), dtype=np.float64)
#     for contour_idx in range(num_contours):
#         y = (height - 1) - contour_idx
#         displacements[contour_idx, :] = red[y, :]

#     return displacements


# def bin_displacements(
#     displacements: np.ndarray,
#     points_per_bin: int,
#     contours_per_bin: int
# ) -> np.ndarray:
#     """
#     Aggregates the displacement array by averaging within bins.
#     - points_per_bin: number of consecutive points to average (x-axis bin)
#     - contours_per_bin: number of consecutive contours to average (y-axis bin)
#     """
#     num_contours, num_points = displacements.shape
#     n_contour_bins = num_contours // contours_per_bin
#     n_point_bins = num_points // points_per_bin

#     # Trim extras so it divides evenly
#     trimmed = displacements[
#         : (n_contour_bins * contours_per_bin),
#         : (n_point_bins * points_per_bin)
#     ]

#     # Reshape: (contour_bins, contours_per_bin, point_bins, points_per_bin)
#     reshaped = trimmed.reshape(
#         n_contour_bins,
#         contours_per_bin,
#         n_point_bins,
#         points_per_bin
#     )

#     # Average over the small bins
#     binned = reshaped.mean(axis=(1, 3))  # collapse the two inner axes
#     return binned  # shape = (n_contour_bins, n_point_bins)

# def plot_binned_heatmap(
#     binned_disp: np.ndarray,
#     title: str = "Binned Displacement Heatmap"
# ) -> None:
#     """
#     Plots the binned displacement data as a heatmap,
#     with contour bins on the x-axis and point bins on the y-axis.
#     """
#     plt.figure(figsize=(8, 6))
#     # Transpose so that x-axis = contour bins, y-axis = point bins
#     plt.imshow(binned_disp.T, origin='lower', aspect='auto')
#     plt.colorbar(label='Normalized Displacement')
#     plt.title(title)
#     plt.xlabel('Contour Bin Index')
#     plt.ylabel('Point Bin Index')
#     plt.tight_layout()
#     plt.show()


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


# def plot_heatmap_and_paired_violins_or_box(
#     phase: str,
#     disp_df: pd.DataFrame,
#     meas_df: pd.DataFrame,
#     n_circumferential_bins: int = 10,
#     n_radial_bins: int = 36,
#     cmap_heat="viridis",
#     box_plot: bool = False
# ) -> None:
#     """
#     Draws a figure with two rows:
#       • Top row: heatmap of mean displacement for `phase`.
#       • Bottom row: for each of the 10 contour-bins (bin_0..bin_9), a pair of violin plots:
#             - lumen_area_dia (blue, shade based on bin index)
#             - lumen_area_sys (red, shade based on bin index)

#     Parameters
#     ----------
#     phase : str
#         Name of the phase (e.g., "rest", "stress", "dia_dia", "sys_sys").
#     disp_df : pandas.DataFrame
#         The displacement-map for this phase, shape (n_radial_bins, n_circumferential_bins),
#         with columns ["bin_0", …, "bin_9"] and index 0..(n_radial_bins-1).
#     meas_df : pandas.DataFrame
#         The measurement-map for this phase, **long-form**, with columns:
#             ["bin", "lumen_area_dia", "lumen_area_sys"].
#         Each row is one patientxbin.  "bin" is a string "bin_0"… "bin_9".
#     n_circumferential_bins : int
#         How many contour-bins (default 10).
#     n_radial_bins : int
#         How many radial rows in the heatmap (default 36).
#     cmap_heat : str
#         Matplotlib colormap name for the heatmap (e.g. "viridis").

#     You must ensure disp_df has exactly columns ["bin_0"… "bin_{n_circumferential_bins-1}"].
#     You must ensure meas_df is filtered so that `meas_df["bin"]` only goes from "bin_0"… "bin_{n_circumferential_bins-1}".
#     """

#     # 1) Prepare the color lists for the bottom violins:
#     blues = plt.get_cmap("Blues")
#     reds = plt.get_cmap("Reds")
#     sample_pts = np.linspace(0.3, 0.8, n_circumferential_bins)
#     dia_colors = [blues(x) for x in sample_pts]
#     sys_colors = [reds(x) for x in sample_pts]
#     # Custom x labels: 0=0-10°, 1=10-20°, ..., 10=100-110°
#     y_labels = [f"{i*10}-{(i+1)*10}°" for i in range(36)]
#     # Custom y labels for bins
#     x_labels = [
#         "IM 0-20%", "IM 20-40%", "IM 40-60%", "IM 60-80%", "IM 80-100%",
#         "EM 100-120%", "EM 120-140%", "EM 140-160%", "EM 160-180%", "EM 180-200%",
#     ]

#     # 2) Set up figure with two rows, shared x‐axis for the violins (bins 0..n_circ‐1)
#     fig = plt.figure(constrained_layout=True, figsize=(10, 12))
#     gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

#     # Top: heatmap
#     ax_heat = fig.add_subplot(gs[0, 0])
#     sns.heatmap(
#         disp_df,
#         ax=ax_heat,
#         cmap=cmap_heat,
#         cbar_kws={"label": "Mean displacement"},
#         xticklabels=x_labels,
#         yticklabels=y_labels  # we'll set our own y‐labels if needed
#     )
#     ax_heat.set_title(f"{phase.replace('_',' ').title()} ― Mean Displacement")
#     ax_heat.set_xlabel("Contour bin (0…9)")
#     ax_heat.set_ylabel("Radial bins (0…{})".format(n_radial_bins - 1))
#     ax_heat.set_xticks(ticks=np.arange(len(disp_df.columns)) + 0.5, labels=x_labels, rotation=45)
#     ax_heat.set_yticks(ticks=np.arange(len(disp_df.index)) + 0.5, labels=y_labels, rotation=0)
#     # Invert x so that bin_0 is on the left:
#     ax_heat.invert_xaxis()

#     # Bottom: paired violins
#     ax_violin = fig.add_subplot(gs[1, 0], sharex=ax_heat)

#     # Convert wide-form meas_df to long-form
#     meas_long = meas_df.melt(
#         id_vars=["bin"],
#         value_vars=["lumen_area_dia", "lumen_area_sys"],
#         var_name="measurement",
#         value_name="lumen_area"
#     )
#     # Extract bin index (0-9)
#     meas_long["bin_idx"] = meas_long["bin"].str.replace("bin_", "").astype(int)

#     if box_plot:
#         sns.boxplot(
#             x="bin_idx",
#             y="lumen_area",
#             hue="measurement",
#             data=meas_long,
#             palette={"lumen_area_dia": "blue", "lumen_area_sys": "red"},
#             ax=ax_violin,
#             linewidth=0.5,
#             # fliersize=0  # Hide outliers
#         )
#     else:
#         # Plot violins with hue split
#         sns.violinplot(
#             x="bin_idx",
#             y="lumen_area",
#             hue="measurement",
#             data=meas_long,
#             split=False,  # Side-by-side violins
#             palette={"lumen_area_dia": "blue", "lumen_area_sys": "red"},
#             inner="quartile",
#             ax=ax_violin,
#             bw=0.2,
#             cut=0,
#             linewidth=0.5
#         )

#     # Tidy up
#     ax_violin.set_title(f"{phase.replace('_',' ').title()} ― Lumen Area per Contour Bin")
#     ax_violin.set_xlabel("Contour bin (°)")
#     ax_violin.set_ylabel("Lumen area")
#     ax_violin.invert_xaxis()

#     # Set x-ticks (0-9 → 0-90°)
#     xticks = np.arange(n_circumferential_bins)
#     ax_violin.set_xticks(xticks)
#     ax_violin.set_xticklabels([f"{i*10}-{(i+1)*10}°" for i in xticks], rotation=45)

#     plt.show()


def plot_heatmap_and_paired_violins_or_box(
    phase: str,
    disp_df: pd.DataFrame,
    meas_df: pd.DataFrame,
    cmap_heat: str = "viridis",
    box_plot: bool = False,
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
    y_labels = [f"{i*10}-{(i+1)*10}°" for i in range(n_radial_bins)]

    # 3) Prepare gradient colors for bottom violins/boxes
    blues = plt.get_cmap("Blues")
    reds = plt.get_cmap("Reds")
    sample_pts = np.linspace(0.3, 0.8, n_circumferential_bins)
    dia_colors = [blues(x) for x in sample_pts]
    sys_colors = [reds(x) for x in sample_pts]

    # 4) Set up figure with two rows
    fig = plt.figure(constrained_layout=True, figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    # ===== TOP: HEATMAP =====
    ax_heat = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        disp_df,
        ax=ax_heat,
        cmap=cmap_heat,
        cbar_kws={"label": "Mean displacement"},
        xticklabels=False,
        yticklabels=False,
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

        dia_vals = sub["lumen_area_dia"].dropna().values
        sys_vals = sub["lumen_area_sys"].dropna().values
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
                    patch.set_facecolor(dia_colors[i])

            if sys_vals.size > 0:
                bp = ax_v.boxplot(
                    sys_vals,
                    positions=[pos_s],
                    widths=0.3,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(sys_colors[i])

        else:
            # ========== VIOLINS ==========
            if dia_vals.size > 0:
                vp = ax_v.violinplot(
                    dia_vals, positions=[pos_d], widths=0.3, showextrema=False
                )
                vp["bodies"][0].set_facecolor(dia_colors[i])
                vp["bodies"][0].set_edgecolor("black")
                vp["bodies"][0].set_alpha(0.8)

            if sys_vals.size > 0:
                vp = ax_v.violinplot(
                    sys_vals, positions=[pos_s], widths=0.3, showextrema=False
                )
                vp["bodies"][0].set_facecolor(sys_colors[i])
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
    mid = n_circumferential_bins // 2
    dia_patch = Patch(facecolor=dia_colors[mid], label="Lumen Area Diastolic")
    sys_patch = Patch(facecolor=sys_colors[mid], label="Lumen Area Systolic")
    ax_v.legend(handles=[dia_patch, sys_patch], loc="upper right")

    plt.show()


if __name__ == "__main__":
    # # === User-adjustable parameters ===
    # FILENAME = "data_eacvi/3d_ivus/NARCO_119/rest/mesh_029_rest.png"
    # N_POINTS = 501
    # POINTS_PER_BIN = 14    # e.g. ~10° per bin if 360°/501 ≈ 0.72°
    # CONTOURS_PER_BIN = 2

    # # Load normalized displacements
    # disp_norm = load_normalized_displacements(FILENAME, N_POINTS)

    # # Bin the data
    # binned = bin_displacements(disp_norm, POINTS_PER_BIN, CONTOURS_PER_BIN)

    # # Plot
    # plot_binned_heatmap(binned, title="Rest Mesh Binned Displacements")
    disp_dict = _create_summary_displacement("data_eacvi/output/patient_stats")
    meas_dict = load_measurement_data("data_eacvi/output/patient_stats")
    plot_all_heatmaps(disp_dict, cmap="coolwarm")
    print(meas_dict)
    plot_heatmap_and_paired_violins_or_box(
        phase="rest",
        disp_df=disp_dict["rest_sys_dia"],
        meas_df=meas_dict["rest"],
        box_plot=True,
    )
    plot_heatmap_and_paired_violins_or_box(
        phase="dia_dia", disp_df=disp_dict["dia_dia"], meas_df=meas_dict["dia_dia"]
    )
