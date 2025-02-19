'''
A script for conversion of the model grid to a new format.

Author: Vitaliy Ogarko
'''

import numpy as np
import sys
import os

#=============================================================================
def main():

    if (len(sys.argv) > 1):
        # Path to input model grid (modelGrid parameter in the Parfile).
        filename = sys.argv[1]
        print(filename)
    else:
        print("Usage: convert_model_grid.py <Model_grid_file_path>")
        exit(0)

    # Extract base filename.
    basename = os.path.basename(filename)
    print(basename)

    #--------------------------------------------------------------------------
    # Read the model values.
    model = np.loadtxt(filename, dtype=float, usecols=(6), skiprows=1)

    Nmodel = model.size

    print("Nmodel =", Nmodel)

    new_model_file = 'new_model_' + basename

    # Save extracted model to file.
    np.savetxt(new_model_file, model, delimiter=' ', fmt="%f", header=str(Nmodel), comments='')

    print("Wrote the model to file:", new_model_file)

    #--------------------------------------------------------------------------
    # Read the model grid.
    grid = np.loadtxt(filename, dtype=float, usecols=(0,1,2,3,4,5), skiprows=1)
    # Extract model dimensions from the last line indexes.
    nxyz = np.loadtxt(filename, dtype=int, usecols=(7,8,9), skiprows=1 + Nmodel - 1)

    nx = nxyz[0]
    ny = nxyz[1]
    nz = nxyz[2]

    assert nx * ny * nz == Nmodel

    print("nx, ny, nz =", nx, ny, nz)

    Xgrid = np.zeros(nx + 1, dtype=float)
    Ygrid = np.zeros(ny + 1, dtype=float)
    Zgrid = np.zeros(nz + 1, dtype=float)
    Ztopo = np.zeros((ny, nx), dtype=float)

    p = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (k == 0 and j == 0):
                    Xgrid[i] = grid[p, 0]
                    Xgrid[i + 1] = grid[p, 1]

                if (k == 0 and i == 0):
                    Ygrid[j] = grid[p, 2]
                    Ygrid[j + 1] = grid[p, 3]

                if (i == 0 and j == 0):
                    Zgrid[k] = grid[p, 4]
                    Zgrid[k + 1] = grid[p, 5]

                if (k == 0):
                    # Store the grid elevation (topography).
                    Ztopo[j, i] = grid[p, 4]

                p = p + 1

    # Shift Zgrid values so that it starts from zero. The topography shifts are provided separately via Ztopo.
    Zgrid = Zgrid - Zgrid[0]

    new_grid_file = 'new_grid_' + basename

    # Save extracted model grid to file.
    with open(new_grid_file, "w") as f:
        np.savetxt(f, nxyz.reshape(1, nxyz.size), fmt='%d')
        np.savetxt(f, Xgrid.reshape(1, Xgrid.size), fmt='%f')
        np.savetxt(f, Ygrid.reshape(1, Ygrid.size), fmt='%f')
        np.savetxt(f, Zgrid.reshape(1, Zgrid.size), fmt='%f')
        f.write("1\n")
        np.savetxt(f, Ztopo, fmt='%f')

    print("Wrote the grid to file:", new_grid_file)

#=============================================================================
if __name__ == "__main__":
    main()

