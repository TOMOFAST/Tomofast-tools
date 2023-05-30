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
    return math.pi/180 * float(deg)

def main(induced:list, remanentv:list, use_old_conversion=False):
    # angles are in deg
    # susceptibility (k) in SI (default 0.01)
    # amplitude in nT
    
    # induced input form: (amplitude, inclination, declination)
    # remenantv input form: list of vectors
    # indiv remenant form: (Q factor, inclination, declination, susceptibility)

    if use_old_conversion:
        convert_geosph2sph(induced)
        ix =  math.sin(convert_deg2rad(induced[1])) * math.cos(convert_deg2rad(induced[2]))
        iy =  math.sin(convert_deg2rad(induced[1])) * math.sin(convert_deg2rad(induced[2]))
        iz =  math.cos(convert_deg2rad(induced[1]))
    else:
        ix =  math.cos(convert_deg2rad(induced[1])) * math.cos(convert_deg2rad(induced[2]))
        iy =  math.cos(convert_deg2rad(induced[1])) * math.sin(convert_deg2rad(induced[2]))
        iz =  math.sin(convert_deg2rad(induced[1]))

    print('Inducing Vector:', iy, ix, -1*iz)

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
            rx = AMPL * math.cos(convert_deg2rad(remanent[1])) * math.cos(convert_deg2rad(remanent[2]))
            ry = AMPL * math.cos(convert_deg2rad(remanent[1])) * math.sin(convert_deg2rad(remanent[2]))
            rz = AMPL * math.sin(convert_deg2rad(remanent[1]))
        
        print(ry, rx, -1*rz, '\n', iy, ix, -1*iz)
        
        
        mx = ix + rx
        my = iy + ry
        mz = iz + rz
        
        mx *= k * induced[0]
        my *= k * induced[0]
        mz *= k * induced[0]

        if use_old_conversion:
            # added conversions so numbers match those inside magnetic_field.f90 
            # swapped my and mx
            # mirrored mz
            print("output form: Mx My Mz", "{} {} {}".format(my, mx, -1*mz), '\n')
        else:
            # mx and my remains as they are
            # mz not mirrored
            print("output form: Mx My Mz", "{} {} {}".format(my, mx, mz), '\n')

if __name__ == '__main__':    
    INDUCED = [55000, -60, 2]
    REMANENTS = [
        [0, -60, 2, 0.01]
    ]

    main(INDUCED, REMANENTS)
