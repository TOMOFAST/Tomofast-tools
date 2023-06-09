from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
import os


@dataclass
class TomofastxSensit:
    nx: int
    ny: int
    nz: int
    compression_type: int
    matrix: object
    weight: object


# =========================================================================================
def load_sensit_from_tomofastx(sensit_path, nbproc, verbose=False):
    """
    Loads the sensitivity kernel from Tomofast-x and stores it in the CSR sparse matrix.
    """

    # Metadata file.
    filename_metadata = sensit_path + "/sensit_grav_" + str(nbproc) + "_meta.dat"
    # Depth weight file.
    filename_weight = sensit_path + "/sensit_grav_" + str(nbproc) + "_weight"

    # ----------------------------------------------------------
    # Reading the metadata.
    with open(filename_metadata, "r") as f:
        lines = f.readlines()

        # Read model dimensions.
        nx = int(lines[0].split()[0])
        ny = int(lines[0].split()[1])
        nz = int(lines[0].split()[2])

        if verbose:
            print('Tomofastx nx, ny, nz =', nx, ny, nz)

        # Reading the number of data.
        ndata_read = int(lines[0].split()[3])

        if verbose:
            print('ndata_read =', ndata_read)

        # Reading the number of procs.
        nbproc_read = int(lines[0].split()[4])

        if (nbproc != nbproc_read):
            raise Exception('Inconsistent nbproc!')

        if verbose:
            print('nbproc_read =', nbproc_read)

        compression_type = int(lines[1].split()[0])

        if verbose:
            print('compression_type =', compression_type)

        if compression_type > 1:
            raise Exception('Inconsistent compression type!')

        # The number of non-zero values.
        # NOTE: It should be lines[1] for Tomofast-x v.1.6
        nel_total = sum(map(int, lines[3].split()))

        if verbose:
            print("nel_total =", nel_total)

    # ----------------------------------------------------------
    # Reading depth weight.
    with open(filename_weight, "r") as f:
        # Note using '>' for big-endian.
        header = np.fromfile(f, dtype='>i4', count=5)
        weight = np.fromfile(f, dtype='>f8', count=nel_total)

    # ----------------------------------------------------------
    # Define spase matrix data arrays.
    csr_dat = np.ndarray(shape=(nel_total), dtype=np.float32)
    csr_row = np.ndarray(shape=(nel_total), dtype=np.int32)
    csr_col = np.ndarray(shape=(nel_total,), dtype=np.int32)

    nel_current = 0
    ndata_all = 0

    # Loop over parallel matrix chunks.
    for n in range(nbproc):

        # Sensitivity kernel file.
        filename_sensit = sensit_path + "/sensit_grav_" + str(nbproc) + "_" + str(n)

        # Building the matrix arrays.
        with open(filename_sensit, "r") as f:
            # Global header.
            header = np.fromfile(f, dtype='>i4', count=5)
            ndata_loc = header[0]
            ndata = header[1]
            nmodel = header[2]

            if verbose:
                print("ndata_loc =", ndata_loc)
                print("ndata =", ndata)
                print("nmodel =", nmodel)

            ndata_all += ndata_loc

            # Loop over matrix rows.
            for i in range(ndata_loc):
                # Local line header.
                header_loc = np.fromfile(f, dtype='>i4', count=4)

                # Global data index.
                idata = header_loc[0]

                # Number of non-zero elements in this row.
                nel = header_loc[1]

                # Reading one matrix row.
                col = np.fromfile(f, dtype='>i4', count=nel)
                dat = np.fromfile(f, dtype='>f4', count=nel)

                # Array start/end indexes corresponding to the current matrix row.
                s = nel_current
                e = nel_current + nel

                csr_col[s:e] = col
                csr_row[s:e] = idata - 1
                csr_dat[s:e] = dat

                nel_current = nel_current + nel
    # ----------------------------------------------------------
    if verbose:
        print('ndata_all =', ndata_all)

    if (ndata_all != ndata_read):
        raise Exception('Wrong ndata value!')

    # Shift column indexes to convert from Fortran to Python array index.
    csr_col = csr_col - 1

    # Convert units from Tomofast to geomos (as we use different gravitational constant).
    csr_dat = csr_dat * 1.e+3

    # Create a sparse matrix object.
    matrix = csr_matrix((csr_dat, (csr_row, csr_col)), shape=(ndata_all, nmodel))

    sensit = TomofastxSensit(nx, ny, nz, compression_type, matrix, weight)

    # Keep minimal verbose.
    print("Sensitivity matrix from Tomofastx: loaded.")

    return sensit


# =========================================================================================
def write_tomofast_model_grid(line_data, output_folder="tomofast_grids"):
    """
    Write Tomofast-x model grid.
    """

    filename = output_folder + "/model_grid.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    num_cells = line_data.shape[0]
    print("num_cells = ", num_cells)

    with open(filename, "w") as file:
        # Write the header.
        file.write("%d\n" % num_cells)

        np.savetxt(file, line_data, fmt="%f %f %f %f %f %f %f %d %d %d")
    file.close()


# =========================================================================================
def write_tomofast_data_grid(data_coords, data_vals, output_folder="tomofast_grids"):
    """
    Write Tomofast-x data grid with values.
    """

    num_data = data_coords.x_data.shape[0]
    print("num_data =", num_data)

    data_array = np.ndarray(shape=(num_data, 4), dtype=float)
    data_array[:, 0] = data_coords.x_data
    data_array[:, 1] = data_coords.y_data
    data_array[:, 2] = data_coords.z_data
    data_array[:, 3] = data_vals

    # Write data grid.
    filename = output_folder + "/data_grid.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as file:
        # Write the header.
        file.write("%d\n" % num_data)

        np.savetxt(file, data_array)
    file.close()


# =========================================================================================
def read_tomofast_data(grav_data, filename, data_type):
    """
    Read data and grid stored in Tomofast-x format.
    """
    data = np.loadtxt(filename, skiprows=1)

    if data_type == 'field':
        grav_data.data_field = data[:, 3]

    elif data_type == 'background':
        grav_data.background = data[:, 3]

    # Reading the data grid.
    grav_data.x_data = data[:, 0]
    grav_data.y_data = data[:, 1]
    grav_data.z_data = data[:, 2]


# =========================================================================================
def read_tomofast_model(filename, mpars):
    """
    Read model values and model grid stored in Tomofast-x format.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

        nelements = int(lines[0].split()[0])

    # Sanity check.
    assert nelements == np.prod(mpars.dim), "Wrong model dimensions in read_tomofast_model!"

    model = np.loadtxt(filename, skiprows=1)

    # Define model values.
    m_inv = model[:, 6]
    m_inv = m_inv.reshape(mpars.dim)

    # Define model grid (cell centers).
    mpars.x = 0.5 * (model[:, 0] + model[:, 1])
    mpars.y = 0.5 * (model[:, 2] + model[:, 3])
    mpars.z = 0.5 * (model[:, 4] + model[:, 5])

    # Convert to km.
    mpars.x = mpars.x / 1000.
    mpars.y = mpars.y / 1000.
    mpars.z = mpars.z / 1000.

    return m_inv, mpars


# =========================================================================================
def Haar3D(s, n1, n2, n3):
    """
    Forward Haar wavelet transform.
    """

    for ic in range(3):
        if ic == 0:
            n_scale = int(np.log(float(n1)) / np.log(2.))
            L = n1
        elif ic == 1:
            n_scale = int(np.log(float(n2)) / np.log(2.))
            L = n2
        else:
            n_scale = int(np.log(float(n3)) / np.log(2.))
            L = n3

        for istep in range(1, n_scale + 1):
            step_incr = 2 ** istep
            ngmin = int(step_incr / 2) + 1
            ngmax = ngmin + int((L - ngmin) / step_incr) * step_incr
            ng = int((ngmax - ngmin) / step_incr) + 1
            step2 = step_incr

            # ---------------------------------------------------
            # Predict.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] - s[il, 0:n2, 0:n3]
                elif ic == 1:
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] - s[0:n1, il, 0:n3]
                else:
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] - s[0:n1, 0:n2, il]

                il = il + step2
                ig = ig + step2

            # ---------------------------------------------------
            # Update.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] + s[ig, 0:n2, 0:n3] / 2.
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] + s[0:n1, ig, 0:n3] / 2.
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] + s[0:n1, 0:n2, ig] / 2.

                il = il + step2
                ig = ig + step2

            # ---------------------------------------------------
            # Normalization.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] * np.sqrt(2.)
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] / np.sqrt(2.)
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] * np.sqrt(2.)
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] / np.sqrt(2.)
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] * np.sqrt(2.)
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] / np.sqrt(2.)

                il = il + step2
                ig = ig + step2


# =========================================================================================
def iHaar3D(s, n1, n2, n3):
    """
    Inverse Haar wavelet transform.
    """

    for ic in range(3):
        if ic == 0:
            n_scale = int(np.log(float(n1)) / np.log(2.))
            L = n1
        elif ic == 1:
            n_scale = int(np.log(float(n2)) / np.log(2.))
            L = n2
        else:
            n_scale = int(np.log(float(n3)) / np.log(2.))
            L = n3

        for istep in reversed(range(1, n_scale + 1)):
            step_incr = 2 ** istep
            ngmin = int(step_incr / 2) + 1
            ngmax = ngmin + int((L - ngmin) / step_incr) * step_incr
            ng = int((ngmax - ngmin) / step_incr) + 1
            step2 = step_incr

            # ---------------------------------------------------
            # Normalization.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] / np.sqrt(2.)
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] * np.sqrt(2.)
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] / np.sqrt(2.)
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] * np.sqrt(2.)
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] / np.sqrt(2.)
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] * np.sqrt(2.)

                il = il + step2
                ig = ig + step2

            # ---------------------------------------------------
            # Update.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] - s[ig, 0:n2, 0:n3] / 2.
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] - s[0:n1, ig, 0:n3] / 2.
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] - s[0:n1, 0:n2, ig] / 2.

                il = il + step2
                ig = ig + step2

            # ---------------------------------------------------
            # Predict.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] + s[il, 0:n2, 0:n3]
                elif ic == 1:
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] + s[0:n1, il, 0:n3]
                else:
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] + s[0:n1, 0:n2, il]

                il = il + step2
                ig = ig + step2
