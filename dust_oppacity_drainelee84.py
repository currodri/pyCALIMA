"""
DUST EFFICIENCIES FROM DRAINE & LEE 1984

This set of tools have been constructed such that the public
tables from B. Draine and H.M. Lee can be read, visualised
and reorganised in look-up tables for RAMSES Dust-RTZ

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import some libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Functions

def plack_averaged_effficiencies(filename):
    """
    This function allows for the construction of a clean and
    nice dataset from the Planck-averaged efficiencies.
    """
    data = {}
    columns = ['T_rad', '<Qem>/a', '<Qabs>/a','<Qpr>/a']

    with open(filename) as f:
        # Begin by reading the header
        dust_type = f.readline()
        print(dust_type,'\n')
        for i in range(0,6):
            hd = f.readline()
            print(hd)
        
        # For each grain size, 37 temperatures are computed
        while True:
            myarray = np.zeros((38,4))
            f.readline() # Blank line
            l = f.readline() # Ignore the name of dust type
            if l == '':
                print('End of file')
                break
            grain_size = f.readline().split('=')[0]
            f.readline() # Column names
            for i in range(0, 38):
                line = f.readline()
                myarray[i,:] = np.fromstring(line, dtype=float, sep=' ')
            data[grain_size] = myarray


    return data,columns
                



        