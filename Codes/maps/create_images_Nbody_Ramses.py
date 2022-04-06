# This script creates 2D images/fields from different fields of the simulations
from mpi4py import MPI
import numpy as np
import sys, os, h5py
import MAS_library as MASL
import camels_library as CL
import pynbody
from michaels_functions import center_and_r_vir, remove_bulk_velocity, read_unit_from_info

###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

#################################### INPUT ##########################################
# data parameters
root_in      = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/Sims/Ramses'
root_out     = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses'
prefix       = 'LH'
sim          = 'Ramses'
realizations = 1000
snapnums     = [6]

# images parameters
grid         = 256 #images will have grid x grid pixels
splits       = 5   #number of divisions along one axis (3 x splits images per sim)

# KDTree parameters
k       = 32 #number of neighbors
threads = 4  #number of openmp threads
verbose = False

# parameters of the density field routine
periodic     = True
x_min, y_min = 0.0, 0.0
tracers      = 1000
r_divisions  = 20
#####################################################################################

# do a loop over the different snapshots
for snapnum in snapnums:

    # find the numbers that each cpu will work with
    numbers = np.where(np.arange(realizations)%nprocs==myrank)[0]

    # do a loop over all realizations
    for i in numbers:

        print(i)

        # read the positions of the particles
        snapshot = '%s/%d/output_%05d'%(root_in, i, snapnum)
        if not(os.path.exists(snapshot)):  continue

        data = pynbody.load(snapshot)
        #print(data.all_keys())
        data.physical_units()
        omega_b, omega_m, unit_l, unit_d, unit_t = read_unit_from_info(data)
        unit_v = unit_l/unit_t
        unit_b = np.sqrt(4*np.pi*unit_d)*unit_v
        print(unit_l, unit_d, unit_t, unit_b)
        print(data.properties.keys())
        Omega_m  = data.properties['omegaM0']
        Omega_l  = data.properties['omegaL0']
        h        = data.properties['h']
        redshift = 1.0/data.properties['a'] - 1.0
        BoxSize  = data.properties['boxsize'].in_units('Mpc')*h*(1.0+redshift) #Mpc/h

        # read the positions of the dark matter particles
        pos = data.dm['pos'].in_units('Mpc')*h*(1.0+redshift) #Mpc/h
        pos = pos.astype(np.float32)
        M   = data['mass'].in_units('Msol')*h #Msun/h
        M   = M.astype(np.float32)
        print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
        print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
        print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))
        print(BoxSize)
        print(Omega_m)

        # get name of output file
        fout1 = '%s/Images_M_%s_%s_%d_z=%.2f.npy'%(root_out,sim,prefix,i,abs(redshift))
        if os.path.exists(fout1):  continue
        if myrank==0:  print('Done reading...')

        # get the radii of the different particles
        R = CL.KDTree_distance(pos, pos, k, BoxSize*(1.0+1e-8), threads, verbose)
        R = R.astype(np.float32) #radii as float32; Mpc/h

        if myrank==0:  print('Done computing radii...')

        # define the arrays containing the maps
        maps_M = np.zeros((splits*3, grid, grid), dtype=np.float32)

        # do a loop over the three different axes
        for axis in [0,1,2]:

            axis_x, axis_y = (axis+1)%3, (axis+2)%3 

            # do a loop over the different slices of each axis
            for j in range(splits):

                #if myrank==0:  print(axis,j)
                print(i,axis,j)

                # get the number of the map
                num = axis*splits + j

                # find the range in the slice
                minimum, maximum = j*BoxSize/splits, (j+1)*BoxSize/splits

                # select the particles in the considered slice
                indexes = np.where((pos[:,axis]>=minimum) & (pos[:,axis]<maximum))[0] 
                pos_    = pos[indexes]
                M_      = M[indexes]
                R_      = R[indexes]
                pos_    = np.ascontiguousarray(pos_[:,[axis_x, axis_y]])

                # project mass into a 2D map
                Mtot = np.zeros((grid,grid), dtype=np.float64)
                MASL.projected_voronoi(Mtot, pos_, M_, R_, x_min, y_min, BoxSize,
                                       tracers, r_divisions, periodic, verbose)

                maps_M[num] = Mtot

        np.save(fout1, maps_M)

