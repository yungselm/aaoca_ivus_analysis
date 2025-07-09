from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

def _plot_pca(eigenvalues_dia_ls, eigenvalues_sys_ls, ids):
        """
        
        """
        _, (ax1, ax2, ax3) = plt.subplots(1,3)

        eigenvalues_dia_arr = np.array(eigenvalues_dia_ls)
        eigenvalues_sys_arr = np.array(eigenvalues_sys_ls)

        eigenvalue_ratio = eigenvalues_sys_arr/eigenvalues_dia_arr

        ax1.plot(ids, eigenvalues_dia_arr[:,0],label='diastolic')
        ax1.plot(ids, eigenvalues_sys_arr[:,0], label='systolic')
        ax1.legend()
        ax1.set_title('PC1')
        ax1.set_xlabel('Contour ID [-]')


        ax2.plot(ids, eigenvalues_dia_arr[:,1], label='diastolic')
        ax2.plot(ids, eigenvalues_sys_arr[:,1], label='systolic')
        ax2.legend()
        ax2.set_title('PC2')
        ax2.set_xlabel('Contour ID [-]')

        ax3.plot(ids, eigenvalue_ratio[:,0], label='PC1')
        ax3.plot(ids, eigenvalue_ratio[:,1], label='PC2')
        ax3.set_title('Eigenvalue Ratio Sys/Dia')
        ax3.set_xlabel('Contour ID [-]')
        ax3.set_ylabel(r'$\lambda$ [-]')
        ax3.legend()
        plt.show()

def _get_ostium_points(points, level=0, tol=1e-5) -> np.ndarray:
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


def plot_comparison_frames(
    rest_diastole_mesh, rest_systole_mesh, stress_diastole_mesh, stress_systole_mesh
) -> None:
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
        plt.scatter(
            _get_ostium_points(rest_diastole_mesh, level=i)[:, 0],
            _get_ostium_points(rest_diastole_mesh, level=i)[:, 1],
            color="darkblue",
            marker="o",
            label="Rest Diastole Mesh",
            s=5,
        )

        # Similarly, plot systole points
        plt.scatter(
            _get_ostium_points(rest_systole_mesh, level=i)[:, 0],
            _get_ostium_points(rest_systole_mesh, level=i)[:, 1],
            color="darkred",
            marker="o",
            label="Rest Systole Mesh",
            s=5,
        )
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(
            f"sys_dia_rest/ostium_points_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.close()

    for i in range(0, (len(stress_diastole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest diastole mesh and catheter points with different markers/colors
        plt.scatter(
            _get_ostium_points(stress_diastole_mesh, level=i)[:, 0],
            _get_ostium_points(stress_diastole_mesh, level=i)[:, 1],
            color="darkblue",
            marker="o",
            label="Stress Diastole Mesh",
            s=5,
        )

        # Similarly, plot systole points
        plt.scatter(
            _get_ostium_points(stress_systole_mesh, level=i)[:, 0],
            _get_ostium_points(stress_systole_mesh, level=i)[:, 1],
            color="darkred",
            marker="o",
            label="Stress Systole Mesh",
            s=5,
        )
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(
            f"sys_dia_stress/ostium_points_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.close()

    for i in range(0, (len(rest_diastole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest diastole mesh and catheter points with different markers/colors
        plt.scatter(
            _get_ostium_points(rest_diastole_mesh, level=i)[:, 0],
            _get_ostium_points(rest_diastole_mesh, level=i)[:, 1],
            color="darkblue",
            marker="o",
            label="Rest Diastole Mesh",
            s=1,
        )

        # Similarly, plot systole points
        plt.scatter(
            _get_ostium_points(stress_diastole_mesh, level=i)[:, 0],
            _get_ostium_points(stress_diastole_mesh, level=i)[:, 1],
            color="darkblue",
            marker="o",
            label="Stress Diastole Mesh",
            s=5,
        )
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(
            f"dia_dia_stress/ostium_points_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.close()

    for i in range(0, (len(rest_systole_mesh) // 501)):
        plt.figure(figsize=(10, 10))
        # For example, plot the rest systole mesh and catheter points with different markers/colors
        plt.scatter(
            _get_ostium_points(rest_systole_mesh, level=i)[:, 0],
            _get_ostium_points(rest_systole_mesh, level=i)[:, 1],
            color="darkred",
            marker="o",
            label="Rest Systole Mesh",
            s=1,
        )

        # Similarly, plot systole points
        plt.scatter(
            _get_ostium_points(stress_systole_mesh, level=i)[:, 0],
            _get_ostium_points(stress_systole_mesh, level=i)[:, 1],
            color="darkred",
            marker="o",
            label="Stress Systole Mesh",
            s=5,
        )
        plt.xlim(0, 10)  # Adjust limits as needed
        plt.ylim(0, 10)  # Adjust limits as needed
        plt.legend()
        plt.savefig(
            f"sys_sys_stress/ostium_points_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.close()
