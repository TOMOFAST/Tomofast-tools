'''
A script for converting the Tomofast-x model to GEOH5 open format (to visualize it in Geoscience ANALYST).

Author: Vitaliy Ogarko
'''

import numpy as np
from geoh5py.objects import Points
from geoh5py.workspace import Workspace

def main():

    # Path to input model grid (modelGrid.grav.file parameter in the Parfile).
    filename_model_grid = '../Tomofast-x/data/gravmag/mansf_slice/true_model_grav_3litho.txt'

    # Path to the output model after inversion.
    filename_model_final = '../Tomofast-x/output/mansf_slice/Voxet/grav_final_voxet_full.txt'

    # Reading the model grid.
    model_grid = np.loadtxt(filename_model_grid, dtype=float, usecols=(0,1,2,3,4,5), skiprows=1)

    # Reading the final model.
    model_values = np.loadtxt(filename_model_final, dtype=float, skiprows=1)

    assert model_grid.shape[0] == model_values.shape[0], "Inconsistent model dimensions!"

    Ncells = model_grid.shape[0]
    print("Ncells =", Ncells)

    print(model_grid.shape)
    print(model_values.shape)

    h5file_path = 'model.geoh5'

    # Create a workspace.
    workspace = Workspace(h5file_path)

    # Positions of the model cell centers.
    positions = np.ndarray((Ncells, 3), dtype=float)

    # Calculate the cell centers.
    positions[:, 0] = (model_grid[:, 0] + model_grid[:, 1]) / 2.
    positions[:, 1] = (model_grid[:, 2] + model_grid[:, 3]) / 2.
    positions[:, 2] = (model_grid[:, 4] + model_grid[:, 5]) / 2.

    # Revert Z-axis.
    positions[:, 2] = -positions[:, 2]

    # Create points.
    # (NOTE: Using points - cell centers - as the BlockModel does not support topography).
    points = Points.create(
        workspace,
        vertices = positions,
        )

    # Add point data.
    points.add_data({"model": {"values": model_values}})

    workspace.close()

    print("Saved the output model to:", h5file_path)

#=========================================================================================
if __name__ == "__main__":
    main()
