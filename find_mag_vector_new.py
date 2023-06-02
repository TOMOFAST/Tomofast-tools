import math

# OLD CONVERSION coordinates assume XYZ system follow:
#   +Y: N
#   -Y: S
#   +X: E
#   -X: W
#   +Z: Up (-declination)
#   -Z: Down (+declination)

def convert_geosph2sph(vec:list):
    # (amplitude, inclination, declination)
    # declination measured in x-y plane, positive in the 1st and 4th quadrants, measured from +y axis
    # inclination measured in y-z plane, positive in the 3rd and 4th quadrants, measured from +y axis

    vec[1] += 90
    vec[2] = -vec[2] + 90

def convert_deg2rad(deg):
    return math.pi / 180. * float(deg)

def to_cartesian(ampl, I, D):
    '''
    Convert a vector to cartesion frame, with Z pointing upwards, X = East, Y = North.
    Input: I, D are inclination and declination in degrees.
    '''
    # Convert degrees to radians.
    I = convert_deg2rad(I)
    D = convert_deg2rad(D)

    x = ampl * math.cos(I) * math.sin(D)
    y = ampl * math.cos(I) * math.cos(D)
    z = ampl * math.sin(I)

    return (x, y, z)

def main(induced:list, remanentv:list, use_old_conversion=False):
    # angles are in deg
    # susceptibility (k) in SI (default 0.01)
    # amplitude in nT
    
    # induced input form: (amplitude, inclination, declination)
    # remenantv input form: list of vectors
    # indiv remenant form: (Q factor, inclination, declination, susceptibility)
    print('use_old_conversion:', use_old_conversion)

    if use_old_conversion:
        convert_geosph2sph(induced)
        ix =  math.sin(convert_deg2rad(induced[1])) * math.cos(convert_deg2rad(induced[2]))
        iy =  math.sin(convert_deg2rad(induced[1])) * math.sin(convert_deg2rad(induced[2]))
        iz =  math.cos(convert_deg2rad(induced[1]))
    else:
        ix, iy, iz = to_cartesian(1., induced[1], induced[2])

    print('Inducing Vector:', iy, ix, iz)

    for remanent in remanentv:
        if len(remanent) == 4:
            k = remanent.pop(-1)
        else:
            k = 0.01

        print("Q: {} INCL: {}deg DECL: {}deg SUSC: {}SI".format(*remanent, k))

        AMPL = remanent[0] #* induced[0]

        if use_old_conversion:
            convert_geosph2sph(remanent)
            rx = AMPL * math.sin(convert_deg2rad(remanent[1])) * math.cos(convert_deg2rad(remanent[2]))
            ry = AMPL * math.sin(convert_deg2rad(remanent[1])) * math.sin(convert_deg2rad(remanent[2]))
            rz = AMPL * math.cos(convert_deg2rad(remanent[1]))
        else:
            rx, ry, rz = to_cartesian(AMPL, remanent[1], remanent[2])

        print(ry, rx, rz, '\n', iy, ix, iz)

        mx = ix + rx
        my = iy + ry
        mz = iz + rz

        # Vacuum permeability.
        mu0 = 4. * math.pi * 1.e-7
        # Nanotesla to tesla conversion factor.
        nt2t = 1.e-9

        mx *= k * induced[0] / mu0 * nt2t
        my *= k * induced[0] / mu0 * nt2t
        mz *= k * induced[0] / mu0 * nt2t

        if use_old_conversion:
            # added conversions so numbers match those inside magnetic_field.f90 
            # swapped my and mx
            # mirrored mz
            print("output form: Mx My Mz", "{} {} {}".format(my, mx, -1*mz), '\n')
        else:
            # mx and my remains as they are
            # mz not mirrored
            print("output form: Mx My Mz", "{} {} {}".format(mx, my, mz), '\n')

if __name__ == '__main__':    
    INDUCED = [55000., -60., 2.]
    REMANENTS = [
        [0., -60., 2., 0.01]
    ]

    main(INDUCED, REMANENTS)
