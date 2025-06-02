import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_normalized_displacements(
    filename: str,
    points_per_contour: int
) -> np.ndarray:
    """
    Reads your displacement texture PNG and returns a 2D array of
    shape (num_contours, points_per_contour) with normalized
    displacements in [0.0, 1.0].

    Assumes:
      - width  == points_per_contour
      - height == num_contours
      - R channel = normalized_disp*255
      - Y axis was flipped in Rust:
          y_written = (height - 1) - contour_index
    """
    # load and force RGB
    img = Image.open(filename).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)       # shape = (H, W, 3)
    height, width, _ = arr.shape

    if width != points_per_contour:
        raise ValueError(
            f"Expected width={points_per_contour}, got {width}"
        )

    # Extract red channel and normalize to [0,1]
    red = arr[:, :, 0].astype(np.float64) / 255.0

    # Undo the Y-flip so that contour 0 is row 0, etc.
    num_contours = height
    displacements = np.empty((num_contours, width), dtype=np.float64)
    for contour_idx in range(num_contours):
        y = (height - 1) - contour_idx
        displacements[contour_idx, :] = red[y, :]

    return displacements


def bin_displacements(
    displacements: np.ndarray,
    points_per_bin: int,
    contours_per_bin: int
) -> np.ndarray:
    """
    Aggregates the displacement array by averaging within bins.
    - points_per_bin: number of consecutive points to average (x-axis bin)
    - contours_per_bin: number of consecutive contours to average (y-axis bin)
    """
    num_contours, num_points = displacements.shape
    n_contour_bins = num_contours // contours_per_bin
    n_point_bins = num_points // points_per_bin

    # Trim extras so it divides evenly
    trimmed = displacements[
        : (n_contour_bins * contours_per_bin),
        : (n_point_bins * points_per_bin)
    ]

    # Reshape: (contour_bins, contours_per_bin, point_bins, points_per_bin)
    reshaped = trimmed.reshape(
        n_contour_bins,
        contours_per_bin,
        n_point_bins,
        points_per_bin
    )

    # Average over the small bins
    binned = reshaped.mean(axis=(1, 3))  # collapse the two inner axes
    return binned  # shape = (n_contour_bins, n_point_bins)

def plot_binned_heatmap(
    binned_disp: np.ndarray,
    title: str = "Binned Displacement Heatmap"
) -> None:
    """
    Plots the binned displacement data as a heatmap,
    with contour bins on the x-axis and point bins on the y-axis.
    """
    plt.figure(figsize=(8, 6))
    # Transpose so that x-axis = contour bins, y-axis = point bins
    plt.imshow(binned_disp.T, origin='lower', aspect='auto')
    plt.colorbar(label='Normalized Displacement')
    plt.title(title)
    plt.xlabel('Contour Bin Index')
    plt.ylabel('Point Bin Index')
    plt.tight_layout()
    plt.show()


def _create_summary_displacement(path) -> dict:
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
            "rest":   "rest_sys_dia_displacement_map.csv",
            "stress": "stress_sys_dia_displacement_map.csv",
            "dia_dia": "dia_dia_displacement_map.csv",
            "sys_sys": "sys_sys_displacement_map.csv"
        }

        try:
            df_rest   = pd.read_csv(os.path.join(subpath, filenames["rest"]))
            df_stress = pd.read_csv(os.path.join(subpath, filenames["stress"]))
            df_dia    = pd.read_csv(os.path.join(subpath, filenames["dia_dia"]))
            df_sys    = pd.read_csv(os.path.join(subpath, filenames["sys_sys"]))
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
        "rest_sys_dia":   _average_cellwise_ignore_nan(rest_list),
        "stress_sys_dia": _average_cellwise_ignore_nan(stress_list),
        "dia_dia":        _average_cellwise_ignore_nan(dia_dia_list),
        "sys_sys":        _average_cellwise_ignore_nan(sys_sys_list),
    }


def plot_all_heatmaps(displacement_dict, cmap="viridis"):
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
            yticklabels=df.index
        )
        # Custom x labels: 0=0-10°, 1=10-20°, ..., 10=100-110°
        y_labels = [f"{i*10}-{(i+1)*10}°" for i in range(36)]
        # Custom y labels for bins
        x_labels = [
            "IM 0-20%", "IM 20-40%", "IM 40-60%", "IM 60-80%", "IM 80-100%",
            "EM 100-120%", "EM 120-140%", "EM 140-160%", "EM 160-180%", "EM 180-200%",
        ]
        plt.title(f"{phase.replace('_', ' ').title()} Heatmap")
        plt.xlabel("Vessel position intramural (IM) / extramural (EM)")
        plt.ylabel("Circumferential position (°)")
        plt.xticks(ticks=np.arange(len(df.columns)) + 0.5, labels=x_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(df.index)) + 0.5, labels=y_labels, rotation=0)
        plt.tight_layout()
        plt.gca().invert_xaxis()
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
    result = _create_summary_displacement("data_eacvi/output/patient_stats")
    print(result)
    plot_all_heatmaps(result, cmap="coolwarm")





