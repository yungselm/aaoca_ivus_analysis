# import matplotlib.pyplot as plt
# import numpy as np
# import os

def _get_ostium_points(points, level=0):
    """
    Returns all points with the same index value at the specified level.

    Parameters:
    points (np.ndarray): Array of points (Nx4), columns: [index, x, y, z].
    level (int): The rank of the index to retrieve (0 for highest, 1 for second highest, etc.).

    Returns:
    np.ndarray: Array of points corresponding to the specified index.
    """
    unique_indices = np.unique(points[:, 0])
    sorted_indices = np.sort(unique_indices)

    if level >= len(sorted_indices):
        return np.array([])

    target_index = sorted_indices[-(level + 1)]
    return points[points[:, 0] == target_index]

# def plot_frame_diff(rest_dia, rest_dia_adjusted, rest_sys, rest_sys_adjusted, stress_dia, stress_sys, output_dir):
#     # Create all the output dirs in current dir: 
#     output_dirs = [
#         'sys_dia_rest',
#         'sys_dia_stress',
#         'dia_dia_stress',
#         'sys_sys_stress'
#     ]
#     for d in output_dirs:
#         dir_path = os.path.join(output_dir, d)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)

#     for i in range(0, (len(rest_dia) // 501)):
#         plt.figure(figsize=(10, 10))
#         # For example, plot the rest diastole mesh and catheter points with different markers/colors
#         plt.scatter(_get_ostium_points(rest_dia, level=i)[:, 0], _get_ostium_points(rest_dia, level=i)[:, 1], 
#                     color='darkblue', marker='o', label='Rest Diastole Mesh', s=5)

#         # Similarly, plot systole points
#         plt.scatter(_get_ostium_points(rest_sys, level=i)[:, 0], _get_ostium_points(rest_sys, level=i)[:, 1], 
#                     color='darkred', marker='o', label='Rest Systole Mesh', s=5)
#         plt.xlim(0, 10)  # Adjust limits as needed
#         plt.ylim(0, 10)  # Adjust limits as needed
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'sys_dia_rest', f'ostium_points_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
#         plt.close()

#     for i in range(0, (len(stress_dia) // 501)):
#         plt.figure(figsize=(10, 10))
#         # For example, plot the rest diastole mesh and catheter points with different markers/colors
#         plt.scatter(_get_ostium_points(stress_dia, level=i)[:, 0], _get_ostium_points(stress_dia, level=i)[:, 1], 
#                     color='darkblue', marker='o', label='Stress Diastole Mesh', s=5)

#         # Similarly, plot systole points
#         plt.scatter(_get_ostium_points(stress_sys, level=i)[:, 0], _get_ostium_points(stress_sys, level=i)[:, 1], 
#                     color='darkred', marker='o', label='Stress Systole Mesh', s=5)
#         plt.xlim(0, 10)  # Adjust limits as needed
#         plt.ylim(0, 10)  # Adjust limits as needed
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'sys_dia_stress', f'ostium_points_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
#         plt.close()

#     for i in range(0, (len(rest_dia_adjusted) // 501)):
#         plt.figure(figsize=(10, 10))
#         # For example, plot the rest diastole mesh and catheter points with different markers/colors
#         plt.scatter(_get_ostium_points(rest_dia_adjusted, level=i)[:, 0], _get_ostium_points(rest_dia_adjusted, level=i)[:, 1], 
#                     color='darkblue', marker='o', label='Rest Diastole Mesh', s=1)

#         # Similarly, plot systole points
#         plt.scatter(_get_ostium_points(stress_dia, level=i)[:, 0], _get_ostium_points(stress_dia, level=i)[:, 1], 
#                     color='darkblue', marker='o', label='Stress Diastole Mesh', s=5)
#         plt.xlim(0, 10)  # Adjust limits as needed
#         plt.ylim(0, 10)  # Adjust limits as needed
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'dia_dia_stress', f'ostium_points_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
#         plt.close()    

#     for i in range(0, (len(rest_sys_adjusted) // 501)):
#         plt.figure(figsize=(10, 10))
#         # For example, plot the rest systole mesh and catheter points with different markers/colors
#         plt.scatter(_get_ostium_points(rest_sys_adjusted, level=i)[:, 0], _get_ostium_points(rest_sys_adjusted, level=i)[:, 1], 
#                     color='darkred', marker='o', label='Rest Systole Mesh', s=1)

#         # Similarly, plot systole points
#         plt.scatter(_get_ostium_points(stress_sys, level=i)[:, 0], _get_ostium_points(stress_sys, level=i)[:, 1], 
#                     color='darkred', marker='o', label='Stress Systole Mesh', s=5)
#         plt.xlim(0, 10)  # Adjust limits as needed
#         plt.ylim(0, 10)  # Adjust limits as needed
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'sys_sys_stress', f'ostium_points_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
#         plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

def _get_ostium_points(points, level=0):
    """
    Returns all points with the same index value at the specified level.

    Parameters:
    points (np.ndarray): Array of points (Nx4), columns: [index, x, y, z].
    level (int): The rank of the index to retrieve (0 for highest, 1 for second highest, etc.).

    Returns:
    np.ndarray: Array of points corresponding to the specified index.
    """
    unique_indices = np.unique(points[:, 0])
    sorted_indices = np.sort(unique_indices)

    if level >= len(sorted_indices):
        return np.array([])

    target_index = sorted_indices[-(level + 1)]
    return points[points[:, 0] == target_index]


def plot_frame_diff(rest_dia, rest_dia_adjusted, rest_sys, rest_sys_adjusted, stress_dia, stress_sys, output_dir):
    # Create all the output dirs in the specified output_dir
    output_dirs = [
        'sys_dia_rest',
        'sys_dia_stress',
        'dia_dia_stress',
        'sys_sys_stress'
    ]
    for d in output_dirs:
        path = os.path.join(output_dir, d)
        os.makedirs(path, exist_ok=True)

    # Determine number of frames by unique index count for each dataset
    n_rest = np.unique(rest_dia[:, 0]).size
    n_stress = np.unique(stress_dia[:, 0]).size
    n_rest_adj = np.unique(rest_dia_adjusted[:, 0]).size
    n_stress_sys = np.unique(stress_sys[:, 0]).size
    n_frames = max(n_rest, n_stress, n_rest_adj, n_stress_sys)

    # Loop through each frame level
    for i in range(n_frames):
        # Rest diastole vs rest systole
        pts_d = _get_ostium_points(rest_dia, level=i)
        pts_s = _get_ostium_points(rest_sys, level=i)
        plt.figure()
        if pts_d.size:
            plt.scatter(pts_d[:, 1], pts_d[:, 2], s=5)
        if pts_s.size:
            plt.scatter(pts_s[:, 1], pts_s[:, 2], s=5)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(os.path.join(output_dir, 'sys_dia_rest', f'frame_{i}.png'), dpi=300)
        plt.close()

        # Stress diastole vs stress systole
        pts_d = _get_ostium_points(stress_dia, level=i)
        pts_s = _get_ostium_points(stress_sys, level=i)
        plt.figure()
        if pts_d.size:
            plt.scatter(pts_d[:, 1], pts_d[:, 2], s=5)
        if pts_s.size:
            plt.scatter(pts_s[:, 1], pts_s[:, 2], s=5)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(os.path.join(output_dir, 'sys_dia_stress', f'frame_{i}.png'), dpi=300)
        plt.close()

        # Rest diastole adjusted vs stress diastole
        pts_d = _get_ostium_points(rest_dia_adjusted, level=i)
        pts_s = _get_ostium_points(stress_dia, level=i)
        plt.figure()
        if pts_d.size:
            plt.scatter(pts_d[:, 1], pts_d[:, 2], s=5)
        if pts_s.size:
            plt.scatter(pts_s[:, 1], pts_s[:, 2], s=5)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(os.path.join(output_dir, 'dia_dia_stress', f'frame_{i}.png'), dpi=300)
        plt.close()

        # Rest systole adjusted vs stress systole
        pts_d = _get_ostium_points(rest_sys_adjusted, level=i)
        pts_s = _get_ostium_points(stress_sys, level=i)
        plt.figure()
        if pts_d.size:
            plt.scatter(pts_d[:, 1], pts_d[:, 2], s=5)
        if pts_s.size:
            plt.scatter(pts_s[:, 1], pts_s[:, 2], s=5)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(os.path.join(output_dir, 'sys_sys_stress', f'frame_{i}.png'), dpi=300)
        plt.close()
