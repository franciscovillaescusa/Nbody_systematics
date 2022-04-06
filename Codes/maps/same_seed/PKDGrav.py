# This script creates 2D images/fields from different fields of the simulations
import numpy as np
import sys, os, h5py
import MAS_library as MASL
import camels_library as CL
import units_library as UL


rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3

def read_tipsy(name, offset=0, count=-1):
    """
    Reads out particles from a Tipsy snapshot file
    :param name: Path to the snapshot file
    :param offset: Skip this many particles at the beginning
    :param count: Read this many particles, -1 -> read all particles
    """
    with open(name, "rb") as f:

        # read header
        p_header_dt = np.dtype([('a','>d'),('npart','>u4'),('ndim','>u4'),('ng','>u4'),
                                ('nd','>u4'),('ns','>u4'),('buffer','>u4')])
        p_header = np.fromfile(f, dtype=p_header_dt, count=1, sep='')

        # get the total number of particles
        n_part = ((p_header["buffer"] & 0x000000ff).astype(np.uint64) << 32)[0] + p_header["npart"][0]

        # read particle properties
        #p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),
        #                 ("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
        p_dt = np.dtype([('mass','>f'), ("pos",('>f',3)), ("vel",('>f',3)),
                         ("eps",'>f'), ("phi",'>f')])
        count = n_part-int(offset) if count == -1 else count
        p = np.fromfile(f, dtype=p_dt, count=int(count), sep='', offset=offset*p_dt.itemsize)

    return n_part, p_header, p

#################################### INPUT ##########################################
# data parameters
snapshot     = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/Sims/PKDGrav/comparison/1/run.00100'
fout         = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/comparison/Images_M_PKDGrav_fiducial_z=0.00.npy'

# images parameters
grid         = 256 #images will have grid x grid pixels
splits       = 5   #number of divisions along one axis (3 x splits images per sim)

# KDTree parameters
k       = 32  #number of neighbors
threads = 5   #number of openmp threads
verbose = False

# parameters of the density field routine
periodic     = True
x_min, y_min = 0.0, 0.0
tracers      = 1000
r_divisions  = 20
#####################################################################################


# read the positions of the particles
BoxSize = 25.0 #Mpc/h
npart, header, p = read_tipsy(snapshot)
pos  = (p['pos']+0.5)*BoxSize  #Mpc/h
vel  = p['vel']*100.0*BoxSize*np.sqrt(3.0/(8.0*np.pi)) #km/s
M    = p['mass']*BoxSize**3*rho_crit #Msun/h
Omega_m = np.sum(M, dtype=np.float64)/BoxSize**3/rho_crit
print('scale factor: %.3f'%(header['a']))
print('redshift:     %.3f'%(1.0/header['a']-1))
print('particles:    %d'%npart)
print('Omega_m:      %.4f'%Omega_m)
print('%.3f < X < %.3f'%(np.min(pos[:,0]), np.max(pos[:,0])))
print('%.3f < Y < %.3f'%(np.min(pos[:,1]), np.max(pos[:,1])))
print('%.3f < Z < %.3f'%(np.min(pos[:,2]), np.max(pos[:,2])))

# get name of output file
if os.path.exists(fout):  sys.exit()
print('Done reading...')

# get the radii of the different particles
R = CL.KDTree_distance(pos, pos, k, BoxSize*(1.0+1e-8), threads, verbose)
R = R.astype(np.float32) #radii as float32; Mpc/h
print('Done computing radii...')

# define the arrays containing the maps
maps_M = np.zeros((splits*3, grid, grid), dtype=np.float32)

# do a loop over the three different axes
for axis in [0,1,2]:

    axis_x, axis_y = (axis+1)%3, (axis+2)%3 

    # do a loop over the different slices of each axis
    for j in range(splits):

        #if myrank==0:  print(axis,j)
        print(axis,j)

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

np.save(fout, maps_M)

