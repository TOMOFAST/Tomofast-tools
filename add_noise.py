import random, sys

"""
this file is meant to add noise to tomofast _calc_.txt forward calculations for inversion use
the file should be in the same folder as this python script, otherwise provide its full path

it will output 2 files with appended suffixes to the original file youve provided:
*_withNoise.txt: the file with random noise added to the original reading
*_noiseOnly.txt: a file with the random noise calculated at each point for sanity checking

The coordinates of the observations are untouched
The code is valif for 3 component data
The code assumes the target file has the extension .txt
"""

FORMAT = "{:>15.7f}"

def main(fn, floor, percent):
    """
    floor (float): the lowest absolute value the noise can take; if the random noise is below this level,
    it is set to this level.
    percent (float): the highest absolute value the noise can take - the actual noise added is modified 
    by a multiplier part of random.uniform(-1.,1.)
    """
    
    floor = float(floor)
    percent = float(percent) # percent is in decimals; 1% -> 0.01
    
    fn_out = fn.split('.')[0]+'_withNoise.txt'
    fn_noise = fn.split('.')[0]+'_noiseOnly.txt'
    
    with open(fn) as f_in:
        with open(fn_out, 'w+') as f_out:
            with open(fn_noise, 'w+') as f_noise:
                # write out nEle
                nel = f_in.readline().strip()
                f_out.write(nel)
                f_noise.write(nel)
            
                for line in f_in:
                    if not line.strip():
                        continue
                    
                    line = [float(i.strip()) for i in line.split() if i.split()]
                    
                    # add noise to any components after the 3rd column
                    temp_final = []
                    temp_noise = []
                    for val in line[3:]:
                        noise = val*percent * random.uniform(-1.0,1.0)
                        
                        if abs(noise) < floor: 
                            noise = floor * abs(noise)/noise
                    
                        val += noise
                        
                        temp_final.append(val)
                        temp_noise.append(noise)
                
                    use_format = '\n'+" ".join([FORMAT]*(len(temp_final)+3))
                    f_out.write(use_format.format(*(line[:3]+temp_final)))
                    f_noise.write(use_format.format(*(line[:3]+temp_noise)))
                    

if __name__ == '__main__':
    main(*sys.argv[1:])
                    
