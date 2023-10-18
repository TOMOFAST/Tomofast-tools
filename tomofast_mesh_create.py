import os
import numpy as np
# WAXI Course on inversion with Tomofast-x / Course WAXI sur l'inversion avec Tomofast-x.
# Script to create a mesh for inversion using Tomofast-x.
# Script pour créeun une grille pour l'inversion avec Tomofaast-x.

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


# The UTM coordinates for your mesh to define the area (values in meters).
# Coordonnées UTM de la grille pour l'inversion (valeurs en metres).
x_min = 1000.
y_min = 2000.
z_min = 0.
x_max = 1100.
y_max = 2100.
z_max = 100.

# The number of cells in each direction (dimension of your mesh).
# Nombre de cellules dans chaque direction (dimension de la grille).
nx = 9
ny = 10
nz = 11

# Making the mesh (nothing to change beyond this point).
# Creation de la grille (rien à changer à partir de ce point).
# ===============================================================
nx += 1
ny += 1
nz += 1

x_vect = np.linspace(x_min, x_max, nx)
y_vect = np.linspace(y_min, y_max, ny)
z_vect = np.linspace(z_min, z_max, nz)

dx = (x_max-x_min)/(nx-1)
dy = (y_max-y_min)/(ny-1)
dz = (z_max-z_min)/(nz-1)

i_indices = np.arange(1, nx+1)
j_indices = np.arange(1, ny+1)
k_indices = np.arange(1, nz+1)

Y, Z, X = np.meshgrid(y_vect[:-1], z_vect[:-1], x_vect[:-1])
J_indices, K_indices, I_indices = np.meshgrid(j_indices[:-1], k_indices[:-1], i_indices[:-1])

np.shape(Y)

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
values = np.zeros_like(Z)

X1 = X - dx/2
X2 = X + dx/2
Y1 = Y - dy/2
Y2 = Y + dy/2
Z1 = Z - dz/2
Z2 = Z + dz/2

X1 = X1 - np.min(X1) + x_min
X2 = X2 - np.min(X2) + x_min + dx
Y1 = Y1 - np.min(Y1) + y_min
Y2 = Y2 - np.min(Y2) + y_min + dy
Z1 = Z1 - np.min(Z1) + z_min
Z2 = Z2 - np.min(Z2) + z_min + dz

line_data = np.array([X1, X2, Y1, Y2, Z1, Z2, values,
                      I_indices.flatten(), J_indices.flatten(), K_indices.flatten()]).T

# Writing the file.
write_tomofast_model_grid(line_data)

