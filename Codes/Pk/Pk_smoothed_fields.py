import numpy as np
import sys,os
import MAS_library as MASL
import Pk_library as PKL
import readgadget
import pynbody
import h5py, yt
import units_library as UL
from michaels_functions import remove_bulk_velocity, read_unit_from_info
import asdf
import smoothing_library as SL


def Pk(grid, pos, BoxSize, MAS, W_k, threads, fout1, fout2):

    # compute density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=None, verbose=True)
    #delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

    # smooth the density field
    delta_smoothed = SL.field_smoothing(delta, W_k, threads)

    print('%.5e'%np.mean(delta))
    print('%.5e'%np.mean(delta_smoothed))

    delta          = np.log10(1.0+delta)
    delta_smoothed = np.log10(1.0+delta_smoothed)

    # compute Pk
    Pk = PKL.Pk(delta, BoxSize, axis, MAS=MAS, threads=threads, verbose=True)
    np.savetxt(fout1, np.transpose([Pk.k3D, Pk.Pk[:,0]]))
    Pk = PKL.Pk(delta_smoothed, BoxSize, axis, MAS=MAS, threads=threads, verbose=True)
    np.savetxt(fout2, np.transpose([Pk.k3D, Pk.Pk[:,0]]))


#################################### INPUT ##########################################
root     = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/Sims/same_seed'
root_out = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/PUBLIC/Results/Pk'

# Gadget
snap1  = '%s/Gadget/snap_033.hdf5'%root
fout1  = 'Pk_Gadget.txt'
fout11 = 'Pk_Gadget_smoothed.txt'

# Ramses
snap2  = '%s/Ramses/output_00035'%root
fout2  = 'Pk_Ramses.txt'
fout22 = 'Pk_Ramses_smoothed.txt'

# Pk parameters
grid    = 512
MAS     = 'CIC'
axis    = 0

# smoothing parameters
BoxSize = 25.0 #Mpc/h
R       = BoxSize/grid*6  #Mpc/h
Filter  = 'Gaussian'
threads = 1
#####################################################################################

# compute FFT of the filter
W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

# Gadget
header  = readgadget.header(snap1)
BoxSize = header.boxsize/1e3  #Mpc/h
pos     = readgadget.read_block(snap1, "POS ", [1])/1e3 #positions in Mpc/h
Pk(grid, pos, BoxSize, MAS, W_k, threads, fout1, fout11)

# Ramses
data = pynbody.load(snap2)
data.physical_units()
omega_b, omega_m, unit_l, unit_d, unit_t = read_unit_from_info(data)
h        = data.properties['h']
redshift = 1.0/data.properties['a'] - 1.0
BoxSize  = data.properties['boxsize'].in_units('Mpc')*h*(1.0+redshift) #Mpc/h
pos = data.dm['pos'].in_units('Mpc')*h*(1.0+redshift) #Mpc/h
pos = pos.astype(np.float32)
Pk(grid, pos, BoxSize, MAS, W_k, threads, fout2, fout22)
