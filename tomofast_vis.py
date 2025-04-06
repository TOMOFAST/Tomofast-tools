'''
A script for visualisation of Tomofast-x final model (using Python tools).

Author: Vitaliy Ogarko
'''

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import cm

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.colorbar as cbar

#==================================================================================================
# Visualisation of a 2D model slice.
def draw_model(grid, model, title, palette):
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
    cmap = pl.get_cmap(palette)

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

#==================================================================================================
# Visualisation of forward 1D data profile (along the model slice).
def draw_data(data_obs, data_calc, profile_coord):
    '''
    Draw the data.

    profile_coord = 0, 1, 2, for x, y, z profiles.
    '''
    # Increasing the figure size.
    #pl.figure(figsize=(12.8, 9.6))

    pl.plot(data_obs[:, profile_coord], data_obs[:, 3], '--bs', label='Observed data')
    pl.plot(data_calc[:, profile_coord], data_calc[:, 3], '--ro', label='Calculated data')

    pl.legend(loc="upper left")

    pl.show()
    pl.close(pl.gcf())

#==================================================================================================
# Visualisation of a 3D model.
def plot_3D_model(model, threshold, dzyx, filename="density", top_view=False, title=''):
    model = model.T  # transpose to match plotting orientation
    L, W, H = model.shape

    # Threshold mask
    filled = (abs(model) >= threshold)

    # Color and edgecolor arrays
    facecolors = np.empty(model.shape, dtype=object)
    edgecolors = np.empty(model.shape, dtype=object)

    # Color map setup
    norm = colors.Normalize(vmin=-1.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    # Assign colors to voxels above threshold
    for i, j, k in zip(*np.where(filled)):
        rgba = mapper.to_rgba(model[i, j, k])
        facecolors[i, j, k] = colors.rgb2hex(rgba)
        edgecolors[i, j, k] = '#000000'  # black edge (or rgba if desired)

    # Add dummy invisible voxels at 8 corners to enforce full bounding box
    corners = [
        (0, 0, 0),
        (L-1, 0, 0),
        (0, W-1, 0),
        (0, 0, H-1),
        (L-1, W-1, 0),
        (L-1, 0, H-1),
        (0, W-1, H-1),
        (L-1, W-1, H-1),
    ]
    for i, j, k in corners:
        filled[i, j, k] = True
        facecolors[i, j, k] = '#00000000'        # fully transparent face
        edgecolors[i, j, k] = '#00000000'        # fully transparent edge

    # Call the plotter
    plt_model_3D(filled, facecolors, dzyx, filename, top_view, edgecolors=edgecolors, title=title)

#==================================================================================================
# Visualisation of a 3D model (called by plot_3D_model).
def plt_model_3D(filled, facecolors, dzyx, filename="density", top_view=False, edgecolors=None, title=''):
    fig = pl.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    # View settings
    ax.view_init(45, -45)
    if top_view:
        ax.set_proj_type('ortho')
        ax.view_init(90, -90)

    # Voxel grid coordinates
    x, y, z = np.indices(np.array(filled.shape) + 1)
    x = x * dzyx[2]
    y = y * dzyx[1]
    z = z * dzyx[0]

    # Plot voxels
    ax.voxels(x, y, z, filled, facecolors=facecolors, edgecolors=edgecolors, shade=False)

    # Axis formatting
    pl.axis('scaled')
    ax.invert_zaxis()
    ax.set_xlabel('X', labelpad=2)
    ax.set_ylabel('Y', labelpad=2)
    ax.set_zlabel('Z', labelpad=2)
    
    pl.title(title)
    pl.show()

#==================================================================================================
# Visualisation of forward 2D data.
def plot_field(field, title):
    pl.figure(figsize=(6, 6), dpi = 150)
    pl.imshow(field, cmap="jet", origin='lower')
    pl.colorbar()
    pl.title(title)
    pl.show()

#=====================================================================================================
def main(filename_model_grid, filename_model_final, filename_data_observed, filename_data_calculated,
        slice_index=1, slice_dim=0, palette='viridis', draw_true_model=True):
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

    # Extract the 2D profile.
    slice_filter = (model_indexes[:, slice_dim] == slice_index)

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

    # Remove not-needed columns.
    if (slice_dim == 0):
        model_grid_slice_2d = np.delete(model_grid_slice, [0, 1, 6], axis=1)
    elif (slice_dim == 1):
        model_grid_slice_2d = np.delete(model_grid_slice, [2, 3, 6], axis=1)
    elif (slice_dim == 2):
        model_grid_slice_2d = np.delete(model_grid_slice, [4, 5, 6], axis=1)

    model_final_slice = model_final[slice_filter]

    #----------------------------------------------------------------------------------
    # Drawing the model.
    #----------------------------------------------------------------------------------
    grid = model_grid_slice_2d

    if (draw_true_model):
        draw_model(grid, true_model_slice, "True model.", palette)
    draw_model(grid, model_final_slice, "Final model.", palette)

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
    # Choose coordinate to be used for 1D data plot (data responce from 2D profile).
    if (slice_dim == 0):
        # YZ profile.
        profile_coord = 1
    elif (slice_dim == 1):
        # XZ profile.
        profile_coord = 0
    else:
        # A 2D data responce - not supported here.
        profile_coord = 0

    draw_data(data_observed_slice, data_calculated_slice, profile_coord)

#=============================================================================
if __name__ == "__main__":
    main()
