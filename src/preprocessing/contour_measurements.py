import numpy as np
import pandas as pd
import math

def compute_contour_properties(arr: np.ndarray) -> np.ndarray:
    """ Input: arr with shape (n, 4): (idx, x, y, z) with several contours.
    Output: arr with shape (m, 5): (idx, z, area, min_dist, max_dist, elliptic_ratio)."""
    results = []

    for contour_id in np.unique(arr[:, 0]):
        points = arr[arr[:, 0] == contour_id][:, 1:3]  # (x, y)
        z = arr[arr[:, 0] == contour_id][0, 3]  # z value (assumed constant for the contour)
        n = len(points)
        if n < 3:
            continue  # Skip invalid contours

        # 1) Compute centroid
        cx, cy = points.mean(axis=0)

        # 2) Compute area using the shoelace formula
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        # 3) Compute angles from centroid
        thetas = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        thetas[thetas < 0] += 2 * np.pi

        # 4) Brute-force search for shortest diameter
        min_dist = np.inf
        max_dist = 0.0

        for i in range(n):
            target_angle = (thetas[i] + np.pi) % (2 * np.pi)
            angle_diffs = np.abs(thetas - thetas[i])
            angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
            opposite_diffs = np.abs(angle_diffs - np.pi)

            j = np.argmin(opposite_diffs)
            if j == i:
                continue  # Skip same point

            # Distance between i and j
            dx, dy = points[i] - points[j]
            dist = np.hypot(dx, dy)

            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)

        if min_dist == 0:
            elliptic_ratio = np.nan  # Avoid division by zero
        else:
            elliptic_ratio = max_dist / min_dist

        results.append([contour_id, z, area, min_dist, max_dist, elliptic_ratio])

    return np.array(results, dtype=float)


def calculate_displacement_map(
        arr1: np.ndarray, 
        arr2: np.ndarray, 
        im_length: float, 
        pressure_change: float,
        time_change: float = np.nan,
        output_path: str = "displacement_map.csv"
    ) -> np.ndarray:
    # 1) compute distances
    xy1 = arr1[:, 1:3]
    xy2 = arr2[:, 1:3]
    distances = np.linalg.norm(xy1 - xy2, axis=1)

    # 2) reshape into (501, num_contours)
    num_contours = distances.size // 501
    displacement_map = distances.reshape(num_contours, 501).T  # (501, num_contours)

    # 3) pad to 504 rows by repeating last row 3×
    last_row = displacement_map[-1:, :]            # shape: (1, num_contours)
    padding  = np.repeat(last_row, 3, axis=0)      # shape: (3, num_contours)
    padded_map = np.vstack((displacement_map, padding))  # (504, num_contours)

    # 4) downsample rows into 36 groups of 14
    group_size = 14
    num_groups = padded_map.shape[0] // group_size  # = 36
    summarized_map = (
        padded_map
        .reshape(num_groups, group_size, num_contours)
        .mean(axis=1)                                 # (36, num_contours)
    )

    # 5) z-coordinates per contour
    z_coords = arr1[::501, 3]

    # 6) build bins of width = im_length / 5, as many as fit up to max(z)
    im_length = float(im_length)
    bin_width = im_length / 5
    num_bins  = math.ceil(z_coords.max() / bin_width)
    edges     = np.arange(num_bins+1) * bin_width
    bins      = np.digitize(z_coords, edges) - 1     # 0-based

    # 7) aggregate each bin
    final_map = np.full((num_groups, num_bins), np.nan)
    for i in range(num_bins):
        mask = (bins == i)
        if mask.any():
            final_map[:, i] = summarized_map[:, mask].mean(axis=1)
    
    if time_change is not np.nan:
        final_map /= (pressure_change * time_change)
    else:
        final_map /= pressure_change

    col_labels = [f"bin_{j}" for j in range(final_map.shape[1])]
    df = pd.DataFrame(final_map, columns=col_labels)
    df.to_csv(output_path, index=False)