'''
Drawing Tomofast-x model profiles and data.

Author: Vitaliy Ogarko
'''

import numpy as np
import matplotlib.pylab as pl

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.colorbar as cbar

#=============================================================================
def main():
    print('Started tomofast_vis.')

    #----------------------------------------------------------------------------------
    # Setting the file paths and constants.
    #----------------------------------------------------------------------------------

    # Model grid dimensions (modelGrid.size parameter in the Parfile).
    nx = 2
    ny = 128
    nz = 32

    # Path to input model grid (modelGrid.grav.file parameter in the Parfile).
    filename_model_grid = '../Tomofast-x/Tomofast-x/data/gravmag/mansf_slice/true_model_grav_3litho.txt'

    # Path to the output model after inversion.
    filename_model_final = '../Tomofast-x/Tomofast-x/output/mansf_slice/Voxet/grav_final_voxet_full.txt'

    #----------------------------------------------------------------------------------
    # Reading data.
    #----------------------------------------------------------------------------------
    nelements = nx * ny * nz

    # Reading the model grid.
    model_grid = np.loadtxt(filename_model_grid, dtype=float, usecols=(0,1,2,3,4,5), skiprows=1)
    model_indexes = np.loadtxt(filename_model_grid, dtype=int, usecols=(7,8,9), skiprows=1)

    # Reverse Z-axis.
    model_grid[:, 4] = - model_grid[:, 4]
    model_grid[:, 5] = - model_grid[:, 5]

    assert nelements == model_grid.shape[0], "Wrong model grid dimensions!"

    # Reading the final model.
    model_final = np.loadtxt(filename_model_final, dtype=float, skiprows=1)

    assert nelements == model_final.shape[0], "Wrong final model dimensions!"

    #----------------------------------------------------------------------------------
    # Extract the model slices.
    #----------------------------------------------------------------------------------
    # Extract the YZ profile.
    nx_slice = 1
    slice_filter = (model_indexes[:, 0] == nx_slice)

    print(slice_filter.shape)

    model_grid_slice = model_grid[slice_filter]
    model_final_slice = model_final[slice_filter]

    nelements_slice = model_grid_slice.shape[0]

    print("nelements_slice =", nelements_slice)

    #----------------------------------------------------------------------------------
    # Drawing the model.
    #----------------------------------------------------------------------------------
    grid = model_grid_slice
    model = model_final_slice

    pl.rcParams["figure.figsize"] = (12.8, 9.6) # Default size = (6.4, 4.8)

    # Makes the same scale for x and y axis.
    pl.axis('scaled')

    x_min = np.min(grid[:, 0])
    x_max = np.max(grid[:, 1])
    y_min = np.min(grid[:, 2])
    y_max = np.max(grid[:, 3])
    z_min = np.min(grid[:, 4])
    z_max = np.max(grid[:, 5])

    print("Model dimensions:", x_min, x_max, y_min, y_max, z_min, z_max)

    # Define figure dimensions.
    # YZ profile.
    pl.xlim(y_min, y_max)
    pl.ylim(z_min, z_max)

    currentAxis = pl.gca()
    currentAxis.set_title("Final model")

    # Gradient palette.
    cmap = pl.get_cmap('viridis')

    patches = []
    color_list = []

    val_min = np.min(model)
    val_max = np.max(model)

    for i in range(nelements_slice):
        # Use YZ-profile.
        x1 = grid[i, 2]
        x2 = grid[i, 3]
        y1 = grid[i, 4]
        y2 = grid[i, 5]

        dx = x2 - x1
        dy = y2 - y1

        # Define the rectangle color.
        val = model[i]
        if (val_max != val_min):
            val_norm = (val - val_min) / (val_max - val_min)
        else:
            val_norm = 0.
        color = cmap(val_norm)

        # Adding rectangle.
        patches.append(Rectangle((x1, y1), dx, dy))
        color_list.append(color)

    # Define patches collection with colormap.
    patches_cmap = ListedColormap(color_list)
    patches_collection = PatchCollection(patches, cmap=patches_cmap)
    patches_collection.set_array(np.arange(len(patches)))

    # Add rectangle collection to the figure.
    currentAxis.add_collection(patches_collection)

    # Show the colorbar.
    cax, _ = cbar.make_axes(currentAxis) 
    cb2 = cbar.ColorbarBase(cax, cmap=cmap)

    pl.show()


#=============================================================================
if __name__ == "__main__":
    main()

