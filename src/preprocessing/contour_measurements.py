from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd


def compute_contour_properties(arr: np.ndarray) -> np.ndarray:
    """Input: arr with shape (n, 4): (idx, x, y, z) with several contours.
    Output: arr with shape (m, 5): (idx, z, area, min_dist, max_dist, elliptic_ratio).
    """
    results = []

    for contour_id in np.unique(arr[:, 0]):
        points = arr[arr[:, 0] == contour_id][:, 1:3]  # (x, y)
        z = arr[arr[:, 0] == contour_id][
            0, 3
        ]  # z value (assumed constant for the contour)
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
            angle_diffs = np.where(
                angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs
            )
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
    output_path: str = "displacement_map.csv",
    adjust_pressure_time: bool = True,
) -> np.ndarray:
    # 1) compute distances
    center = np.array([4.5, 4.5])
    xy1 = arr1[:, 1:3]
    xy2 = arr2[:, 1:3]
    distances = np.linalg.norm(xy1 - xy2, axis=1)

    # Compute the “signed” distances:
    # If arr2’s point is farther from center than arr1’s point, keep positive.
    # Otherwise, make negative.
    d1_center = np.linalg.norm(xy1 - center, axis=1)
    d2_center = np.linalg.norm(xy2 - center, axis=1)
    signs = np.where(d2_center > d1_center, 1.0, -1.0)
    distances = distances * signs  # now 'distances' is signed

    # 2) reshape into (501, num_contours)
    num_contours = distances.size // 501
    displacement_map = distances.reshape(num_contours, 501).T  # (501, num_contours)

    # 3) pad to 504 rows by repeating last row 3×
    last_row = displacement_map[-1:, :]  # shape: (1, num_contours)
    padding = np.repeat(last_row, 3, axis=0)  # shape: (3, num_contours)
    padded_map = np.vstack((displacement_map, padding))  # (504, num_contours)

    # 4) downsample rows into 36 groups of 14
    group_size = 14
    num_groups = padded_map.shape[0] // group_size  # = 36
    summarized_map = padded_map.reshape(num_groups, group_size, num_contours).mean(
        axis=1
    )  # (36, num_contours)

    # 5) z-coordinates per contour
    z_coords = arr1[::501, 3]

    # 6) build bins of width = im_length / 5, as many as fit up to max(z)
    im_length = float(im_length)
    bin_width = im_length / 5
    num_bins = math.ceil(z_coords.max() / bin_width)
    edges = np.arange(num_bins + 1) * bin_width
    bins = np.digitize(z_coords, edges) - 1  # 0-based

    # 7) aggregate each bin
    final_map = np.full((num_groups, num_bins), np.nan)
    for i in range(num_bins):
        mask = bins == i
        if mask.any():
            final_map[:, i] = summarized_map[:, mask].mean(axis=1)

    if adjust_pressure_time:
        if time_change is not np.nan:
            final_map /= pressure_change * time_change
        else:
            final_map /= pressure_change

    col_labels = [f"bin_{j}" for j in range(final_map.shape[1])]
    df = pd.DataFrame(final_map, columns=col_labels)
    df.to_csv(output_path, index=False)


def calculate_measurement_map(
    df: pd.DataFrame,
    phase: str,
    im_length: float,
    pressure_change: float,
    time_change: float = np.nan,
    output_path: str = None,
    adjust_pressure_time: bool = True,
) -> None:
    """
    1) Takes a DataFrame `df` containing at least these columns:
       - "z_value"
       - "lumen_area_dia"
       - "lumen_area_sys"
       - "min_dist_dia"
       - "min_dist_sys"
       - "elliptic_ratio_dia"
       - "elliptic_ratio_sys"
    2) Bins the rows by z_value (bin width = im_length/5), computing one row per bin.
    3) Within each z-bin, computes the mean of each of the six measurement columns.
    4) If adjust_pressure_time==True, divides all of those means by pressure_change
       (and also by time_change if time_change is not NaN).
    5) Saves the resulting (num_bins x 6) table to a CSV named "{phase}_measurement_map.csv".

    The output CSV will have columns:
      ["lumen_area_dia", "lumen_area_sys",
       "min_dist_dia",   "min_dist_sys",
       "elliptic_ratio_dia", "elliptic_ratio_sys"]
    and row-indices ["bin_0", "bin_1", ..., "bin_{num_bins-1}"].
    """

    # 1) Extract z_values and compute bins exactly as in your displacement code.
    z_coords = df["z_value"].values
    im_length = float(im_length)
    bin_width = im_length / 5.0
    num_bins = math.ceil(z_coords.max() / bin_width)
    edges = np.arange(num_bins + 1) * bin_width
    bins = np.digitize(z_coords, edges) - 1  # zero‐based bin index

    # 2) The six measurement columns we want to average per z‐bin:
    if phase == "rest" or phase == "stress":
        meas_cols = [
            "lumen_area_dia",
            "lumen_area_sys",
            "min_dist_dia",
            "min_dist_sys",
            "elliptic_ratio_dia",
            "elliptic_ratio_sys",
        ]
    elif phase == "dia_dia":
        meas_cols = [
            "lumen_area_dia_rest",
            "lumen_area_dia_stress",
            "min_dist_dia_rest",
            "min_dist_dia_stress",
            "elliptic_ratio_dia_rest",
            "elliptic_ratio_dia_stress",
        ]
    elif phase == "sys_sys":
        meas_cols = [
            "lumen_area_sys_rest",
            "lumen_area_sys_stress",
            "min_dist_sys_rest",
            "min_dist_sys_stress",
            "elliptic_ratio_sys_rest",
            "elliptic_ratio_sys_stress",
        ]

    # 3) Prepare an empty array to hold (num_bins × 6) of NaN
    final_map = np.full((num_bins, len(meas_cols)), np.nan)

    # 4) For each bin, compute the mean of each measurement column
    for i in range(num_bins):
        mask = bins == i
        if np.any(mask):
            # slice df down to just rows in bin i, then take mean of each column
            final_map[i, :] = df.loc[mask, meas_cols].mean(axis=0).values

    # 5) If requested, divide by pressure_change (and time_change if provided)
    if adjust_pressure_time:
        if not np.isnan(time_change):
            final_map = final_map / (pressure_change * time_change)
        else:
            final_map = final_map / pressure_change

    # 6) Build a pandas.DataFrame with row labels "bin_0", "bin_1", ...
    col_labels = meas_cols[:]  # same six columns
    row_labels = [f"bin_{j}" for j in range(num_bins)]
    out_df = pd.DataFrame(final_map, index=row_labels, columns=col_labels)

    # 7) Write to CSV
    output_name = f"{phase}_measurement_map.csv"
    full_output_path = os.path.join(output_path, output_name)
    out_df.to_csv(full_output_path, index=True)

    # (Optionally) return the DataFrame in case the caller wants to inspect it directly.
    return out_df
