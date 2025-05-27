import numpy as np
import matplotlib.pyplot as plt


def _get_ostium_points(points, level=0, tol=1e-5):
    """
    Returns the points corresponding to a specific highest z-value level.
    
    Parameters:
    points (np.ndarray): Array of points (Nx3).
    level (int): The rank of the z-level to retrieve (0 for highest, 1 for second highest, etc.).
    tol (float): Tolerance for floating point comparison.
    
    Returns:
    np.ndarray: Array of points corresponding to the specified z-level.
    """
    unique_z_values = np.unique(points[:, 2])  # Get unique z-values
    sorted_z_values = np.sort(unique_z_values)  # Sort them in ascending order

    if level >= len(sorted_z_values):  # If level is out of range, return empty array
        return np.array([])

    target_z = sorted_z_values[-(level + 1)]  # Get the specific z-level
    return points[np.abs(points[:, 2] - target_z) < tol]  # Return points at that level


def plot_comparison_frames(rest_diastole_mesh, rest_systole_mesh, 
                           stress_diastole_mesh, stress_systole_mesh):
    """
    Plots pulsatile and stress-induced lumen deformation.
    
    Parameters:
    rest_diastole_mesh (np.ndarray): Points for rest diastole mesh.
    rest_systole_mesh (np.ndarray): Points for rest systole mesh.
    stress_diastole_mesh (np.ndarray): Points for stress diastole mesh.
    stress_systole_mesh (np.ndarray): Points for stress systole mesh.
    """
    for i in range(0, (len(rest_diastole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest diastole mesh and catheter points with different markers/colors
        plt.scatter(_get_ostium_points(rest_diastole_mesh, level=i)[:, 0], _get_ostium_points(rest_diastole_mesh, level=i)[:, 1], 
                    color='darkblue', marker='o', label='Rest Diastole Mesh', s=5)

        # Similarly, plot systole points
        plt.scatter(_get_ostium_points(rest_systole_mesh, level=i)[:, 0], _get_ostium_points(rest_systole_mesh, level=i)[:, 1], 
                    color='darkred', marker='o', label='Rest Systole Mesh', s=5)
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(f'sys_dia_rest/ostium_points_{i}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    for i in range(0, (len(stress_diastole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest diastole mesh and catheter points with different markers/colors
        plt.scatter(_get_ostium_points(stress_diastole_mesh, level=i)[:, 0], _get_ostium_points(stress_diastole_mesh, level=i)[:, 1], 
                    color='darkblue', marker='o', label='Stress Diastole Mesh', s=5)

        # Similarly, plot systole points
        plt.scatter(_get_ostium_points(stress_systole_mesh, level=i)[:, 0], _get_ostium_points(stress_systole_mesh, level=i)[:, 1], 
                    color='darkred', marker='o', label='Stress Systole Mesh', s=5)
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(f'sys_dia_stress/ostium_points_{i}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    for i in range(0, (len(rest_diastole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest diastole mesh and catheter points with different markers/colors
        plt.scatter(_get_ostium_points(rest_diastole_mesh, level=i)[:, 0], _get_ostium_points(rest_diastole_mesh, level=i)[:, 1], 
                    color='darkblue', marker='o', label='Rest Diastole Mesh', s=1)

        # Similarly, plot systole points
        plt.scatter(_get_ostium_points(stress_diastole_mesh, level=i)[:, 0], _get_ostium_points(stress_diastole_mesh, level=i)[:, 1], 
                    color='darkblue', marker='o', label='Stress Diastole Mesh', s=5)
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(f'dia_dia_stress/ostium_points_{i}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()    

    for i in range(0, (len(rest_systole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest systole mesh and catheter points with different markers/colors
        plt.scatter(_get_ostium_points(rest_systole_mesh, level=i)[:, 0], _get_ostium_points(rest_systole_mesh, level=i)[:, 1], 
                    color='darkred', marker='o', label='Rest Systole Mesh', s=1)

        # Similarly, plot systole points
        plt.scatter(_get_ostium_points(stress_systole_mesh, level=i)[:, 0], _get_ostium_points(stress_systole_mesh, level=i)[:, 1], 
                    color='darkred', marker='o', label='Stress Systole Mesh', s=5)
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(f'sys_sys_stress/ostium_points_{i}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()