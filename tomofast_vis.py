'''
Drawing Tomofast-x model profiles and data.

Author: Vitaliy Ogarko
'''

import numpy as np
import matplotlib.pylab as pl
import matplotlib as mpl

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.colorbar as cbar

def draw_model(grid, model, title):
    '''
    Draw the model slice.
    Note: the input grid should be a 2D grid corresponding to the model slice.
    '''
    nelements = model.shape[0]

    pl.rcParams["figure.figsize"] = (12.8, 9.6) # Default size = (6.4, 4.8)

    # Makes the same scale for x and y axis.
    pl.axis('scaled')

    x_min = np.min(grid[:, 0:2])
    x_max = np.max(grid[:, 0:2])
    y_min = np.min(grid[:, 2:4])
    y_max = np.max(grid[:, 2:4])

    print("Model dimensions:", x_min, x_max, y_min, y_max)

    # Define figure dimensions.
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)

    currentAxis = pl.gca()
    currentAxis.set_title(title)

    # Gradient palette.
    cmap = pl.get_cmap('viridis')

    patches = []
    color_list = []

    val_min = np.min(model)
    val_max = np.max(model)

    for i in range(nelements):
        x1 = grid[i, 0]
        x2 = grid[i, 1]
        y1 = grid[i, 2]
        y2 = grid[i, 3]

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
    # Set the correct colorbar scale.
    norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max)
    cb2 = cbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    pl.show()

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
    model_grid = np.loadtxt(filename_model_grid, dtype=float, usecols=(0,1,2,3,4,5,6), skiprows=1)
    model_indexes = np.loadtxt(filename_model_grid, dtype=int, usecols=(7,8,9), skiprows=1)

    # Revert Z-axis.
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

    model_grid_slice = model_grid[slice_filter]

    # When available the true model is stored in the grid file (7th column).
    true_model_slice = model_grid_slice[:, 6]

    # Remove X-data.
    model_grid_slice = model_grid_slice[:, 2:6]

    model_final_slice = model_final[slice_filter]

    #----------------------------------------------------------------------------------
    # Drawing the model.
    #----------------------------------------------------------------------------------
    grid = model_grid_slice

    draw_model(grid, true_model_slice, "True model.")
    draw_model(grid, model_final_slice, "Final model.")

#=============================================================================
if __name__ == "__main__":
    main()

