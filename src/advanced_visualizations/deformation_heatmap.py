from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    # === User-adjustable parameters ===
    FILENAME = "data_eacvi/3d_ivus/NARCO_119/rest/mesh_029_rest.png"
    N_POINTS = 501
    POINTS_PER_BIN = 14    # e.g. ~10° per bin if 360°/501 ≈ 0.72°
    CONTOURS_PER_BIN = 2

    # Load normalized displacements
    disp_norm = load_normalized_displacements(FILENAME, N_POINTS)

    # Bin the data
    binned = bin_displacements(disp_norm, POINTS_PER_BIN, CONTOURS_PER_BIN)

    # Plot
    plot_binned_heatmap(binned, title="Rest Mesh Binned Displacements")





