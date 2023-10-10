#
# Converts the Noddy grav/mag model to Tomofast-x format for inversion.
# Also writes the corresponding data grid.
#
# Author: Vitaliy Ogarko

import numpy as np
import matplotlib.pyplot as plt

#=============================================================================================
def write_model_grid(filename, nx, ny, nz, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, model_values):
    '''
    Writes the Tomofast-x model grid.
    '''
    # Cell sizes.
    dx = int((Xmax - Xmin) / nx)
    dy = int((Ymax - Ymin) / ny)
    dz = int((Zmax - Zmin) / nz)

    print("dx, dy, dz =", dx, dy, dz)

    nelements = nx * ny * nz

    grid = np.zeros((nelements, 10))
    ind = 0

    for k in range(nz):
        Z1 = Zmin + k * dz
        Z2 = Z1 + dz
        for j in range(ny):
            Y1 = Ymin + j * dy
            Y2 = Y1 + dy
            for i in range(nx):
                X1 = Xmin + i * dx
                X2 = X1 + dx

                grid[ind, 0] = X1
                grid[ind, 1] = X2
                grid[ind, 2] = Y1
                grid[ind, 3] = Y2
                grid[ind, 4] = Z1
                grid[ind, 5] = Z2
                grid[ind, 6] = model_values[k, j, i]

                grid[ind, 7] = i + 1
                grid[ind, 8] = j + 1
                grid[ind, 9] = k + 1

                ind = ind + 1

    # Save model grid to file.
    np.savetxt(filename, grid, delimiter=' ', fmt="%f %f %f %f %f %f %f %d %d %d", header=str(nelements), comments='')

#=====================================================================================================
def write_data_grid(filename, nx, ny, Xmin, Xmax, Ymin, Ymax, elevation):
    '''
    Write data grid in the Tomofast-x format.
    '''
    Ndata = nx * ny
    data_tomo = np.zeros((nx * ny, 4))

    print("Writing the data grid with nx, ny =", nx, ny)
    print("Ndata =", Ndata)

    dx = (Xmax - Xmin) / nx
    dy = (Ymax - Ymin) / ny

    p = 0
    for j in range(ny):
        for i in range(nx):
            x = Xmin + i * dx + dx / 2.
            y = Ymin + j * dy + dy / 2.

            data_tomo[p, 0] = x
            data_tomo[p, 1] = y
            data_tomo[p, 2] = -elevation
            data_tomo[p, 3] = 0.

            p = p + 1

    # Write data to file.
    np.savetxt(filename, data_tomo, delimiter=' ', fmt="%f %f %f %f", header=str(Ndata), comments='')

#=====================================================================================================
def read_noddy_model(filename, nx, ny, nz):
    '''
    Reads the Noddy's model values.
    '''
    def generate_specific_rows(filename, row_indices=[]):
        with open(filename) as f:
            # Using enumerate to track line number.
            for i, line in enumerate(f):
                # If line number is in the row index list, then return that line.
                if i in row_indices:
                    # Remove trailing tab charachter.
                    yield line.rstrip() + "\n"

    model = np.zeros((nz, ny, nx), dtype=float)

    for k in range(nz):
        # Define row indexes for each slice accounting for an empty line between the slices.
        row_indices = range(0 + (ny + 1) * k, ny + (ny + 1) * k)
        gen = generate_specific_rows(filename, row_indices)
        # Read the model slice.
        model_slice = np.loadtxt(gen, delimiter='\t')
        # Sanity check.
        assert(model_slice.shape[0] == ny and model_slice.shape[1] == nx)
        # Store the model slice.
        model[k, :, :] = model_slice.copy()

    return model

#=====================================================================================================
def read_header_dimensions(filename):
    '''
    Reads the model dimensions from the Noddy header file.
    '''
    with open(filename) as f:
        f.readline()
        f.readline()
        dim1 = f.readline()
        dim2 = f.readline()
        dim3 = f.readline()

    nx = int(dim1.split("=")[1].strip())
    ny = int(dim2.split("=")[1].strip())
    nz = int(dim3.split("=")[1].strip())

    return nx, ny, nz

#=============================================================================
def main():

    # Use for density model.
    CONVERT_DENSITY_UNITS = True

    # Noddy model file name.
    model_file = "../Noddy_models/noddy_ellipse_fault/noddy_ellipse_fault_den.dic"

    # Noddy header file name.
    header_file = model_file[0:len(model_file) - 3] + "hdr"

    # Read model dimensions.
    nx, ny, nz = read_header_dimensions(header_file)

    print("Model dimensions:", nx, ny, nz)

    # Model grid cell size.
    cell_size = 100.

    # Data grid elevation.
    elevation = 0.1

    Xmin = 0.
    Ymin = 0.
    Zmin = 0.
    Xmax = cell_size * nx
    Ymax = cell_size * ny
    Zmax = cell_size * nz

    print("Xmin, Xmax =", Xmin, Xmax)
    print("Ymin, Ymax =", Ymin, Ymax)
    print("Zmin, Zmax =", Zmin, Zmax)

    #------------------------------------------------------------
    # Write the data grid.
    #------------------------------------------------------------
    filename = "data_grid.txt"

    write_data_grid(filename, nx, ny, Xmin, Xmax, Ymin, Ymax, elevation)

    #------------------------------------------------------------
    # Read the Noddy model.
    #------------------------------------------------------------
    model_values = read_noddy_model(model_file, nx, ny, nz)

    if CONVERT_DENSITY_UNITS:
        # Convert to density anomalies in kg/m3.
        model_values = model_values * 1.e3

    #---------------------------------------------------------------

    unique_vals = np.unique(model_values)

    print("Number of lithos:", len(unique_vals))
    print("Values:", unique_vals)

    #------------------------------------------------------------
    # Write the model grid.
    #------------------------------------------------------------
    filename = "model_grid.txt"
    write_model_grid(filename, nx, ny, nz, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, model_values)

#=============================================================================
if __name__ == "__main__":
    main()