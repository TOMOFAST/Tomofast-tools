'''
Drawing Tomofast-x model profiles and data.

Author: Vitaliy Ogarko
'''

import numpy as np
import matplotlib.pylab as pl

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

    assert nelements == model_grid.shape[0], "Wrong model grid dimensions!"

    # Reading the final model.
    model_final = np.loadtxt(filename_model_grid, dtype=float, usecols=(0,), skiprows=1)

    assert nelements == model_grid.shape[0], "Wrong fianl model dimensions!"

#=============================================================================
if __name__ == "__main__":
    main()

