'''
A script for visualisation of Tomofast-x final model (using Python tools).

Author: Vitaliy Ogarko
'''

import numpy as np
import matplotlib.pyplot as pl
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

    pl.figure(figsize=(12.8, 9.6))

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

    pl.close(pl.gcf())

def draw_data(data_obs, data_calc, profile_coord):
    '''
    Draw the data.

    profile_coord = 0, 1, 2, for x, y, z profiles.
    '''
    # Increasing the figure size.
    #pl.figure(figsize=(12.8, 9.6))

    pl.plot(data_obs[:, profile_coord], data_obs[:, 3], '--bo', label='Observed data')
    pl.plot(data_calc[:, profile_coord], data_calc[:, 3], '--ro', label='Calculated data')

    pl.legend(loc="upper left")

    pl.show()
    pl.close(pl.gcf())

#=====================================================================================================
def main(filename_model_grid, filename_model_final, filename_data_observed, filename_data_calculated):
    print('Started tomofast_vis.')

    #----------------------------------------------------------------------------------
    # Setting the file paths and constants.
    #----------------------------------------------------------------------------------

    # Path to input model grid (modelGrid.grav.file parameter in the Parfile).
    #filename_model_grid = '../Tomofast-x/data/gravmag/mansf_slice/true_model_grav_3litho.txt'

    # Path to the output model after inversion.
    #filename_model_final = '../Tomofast-x/output/mansf_slice/model/grav_final_model_full.txt'

    # Path to observed data (forward.data.grav.dataValuesFile parameter in the Parfile).
    #filename_data_observed = '../Tomofast-x/output/mansf_slice/data/grav_calc_read_data.txt'

    # Path to calculated data after inversion.
    #filename_data_calculated = '../Tomofast-x/output/mansf_slice/data/grav_calc_final_data.txt'

    #----------------------------------------------------------------------------------
    # Reading data.
    #----------------------------------------------------------------------------------

    # Reading the model grid.
    model_grid = np.loadtxt(filename_model_grid, dtype=float, usecols=(0,1,2,3,4,5,6), skiprows=1)
    model_indexes = np.loadtxt(filename_model_grid, dtype=int, usecols=(7,8,9), skiprows=1)

    # Revert Z-axis.
    model_grid[:, 4] = - model_grid[:, 4]
    model_grid[:, 5] = - model_grid[:, 5]

    # Reading the final model.
    model_final = np.loadtxt(filename_model_final, dtype=float, skiprows=1)

    # Reading data.
    data_observed = np.loadtxt(filename_data_observed, dtype=float, usecols=(0,1,2,3), skiprows=1)
    data_calculated = np.loadtxt(filename_data_calculated, dtype=float, usecols=(0,1,2,3), skiprows=1)

    print("Ndata =", data_observed.shape[0])

    #----------------------------------------------------------------------------------
    # Extract the model slices.
    #----------------------------------------------------------------------------------

    # Extract the YZ profile.
    nx_slice = 1
    slice_filter = (model_indexes[:, 0] == nx_slice)

    model_grid_slice = model_grid[slice_filter]

    # When available the true model is stored in the grid file (7th column).
    true_model_slice = model_grid_slice[:, 6]

    # Grid slice dimensions.
    grid_slice_x_min = np.min(model_grid_slice[:, 0:2])
    grid_slice_x_max = np.max(model_grid_slice[:, 0:2])
    grid_slice_y_min = np.min(model_grid_slice[:, 2:4])
    grid_slice_y_max = np.max(model_grid_slice[:, 2:4])

    print("Grid slice dimenion (X): ", grid_slice_x_min, grid_slice_x_max)
    print("Grid slice dimenion (Y): ", grid_slice_y_min, grid_slice_y_max)

    # Remove grid X-data.
    model_grid_slice_2d = model_grid_slice[:, 2:6]

    model_final_slice = model_final[slice_filter]

    #----------------------------------------------------------------------------------
    # Drawing the model.
    #----------------------------------------------------------------------------------
    grid = model_grid_slice_2d

    draw_model(grid, true_model_slice, "True model.")
    draw_model(grid, model_final_slice, "Final model.")

    #----------------------------------------------------------------------------------
    # Extract data slice.
    #----------------------------------------------------------------------------------
    # Select the data located above the model grid slice.
    data_filter_x = np.logical_and(data_observed[:, 0] >= grid_slice_x_min, data_observed[:, 0] <= grid_slice_x_max)
    data_filter_y = np.logical_and(data_observed[:, 1] >= grid_slice_y_min, data_observed[:, 1] <= grid_slice_y_max)
    data_filter = np.logical_and(data_filter_x, data_filter_y)

    data_observed_slice = data_observed[data_filter, :]
    data_calculated_slice = data_calculated[data_filter, :]

    print("Ndata slice =", data_observed_slice.shape[0])

    #----------------------------------------------------------------------------------
    # Drawing the data.
    #----------------------------------------------------------------------------------
    # YZ profile.
    profile_coord = 1

    draw_data(data_observed_slice, data_calculated_slice, profile_coord)

#=============================================================================
if __name__ == "__main__":
    main()

