import numpy as np

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
