#from mpi4py import MPI
import numpy as np
import sys,os
import density_field_library as DFL
import Pk_library as PKL

###### MPI DEFINITIONS ###### 
#comm   = MPI.COMM_WORLD
#nprocs = comm.Get_size()
#myrank = comm.Get_rank()

####################################### INPUT #########################################
root_out          = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps'
sim               = 'Gaussian'
grid              = 256
BoxSize           = 1000.0 #Mpc/h
Rayleigh_sampling = 0
threads           = 1
verbose           = False
MAS               = 'None'
initial_seed      = 4 #4(Gaussian) #5(Gaussian_NCV) #to generate values of A
num_maps          = 100000
#######################################################################################

root_out = root_out + '/maps_%s'%sim

# initialize random seed and get the values of the parameters and the seeds
# notice that all cpus will have the same values
np.random.seed(initial_seed)
A    = 0.6 + 0.8*np.random.rand(num_maps)
B    = -0.5
seed = np.arange(num_maps)
 
# define the matrix hosting all the maps and power spectra
maps_total   = np.zeros((num_maps, grid, grid), dtype=np.float32)
Pk_total     = np.zeros((num_maps, 181),        dtype=np.float32)

# get the k-bins
k_in = np.logspace(-4,1,500, dtype=np.float32)

# do a loop over all the maps
for i in range(num_maps):#:

    if i%10000==0:  print(i)

    # get the value of the Pk
    Pk_in = A[i]*k_in**B

    # generate density field
    maps_total[i] = DFL.gaussian_field_2D(grid, k_in, Pk_in, Rayleigh_sampling, 
                                            seed[i], BoxSize, threads, verbose)

    # compute power spectrum
    Pk_total[i] = PKL.Pk_plane(maps_total[i], BoxSize, MAS, threads, verbose).Pk

    #Pk_dumb = PKL.Pk_plane(maps_total[i], BoxSize, MAS, threads, verbose)
    #np.savetxt('Gaussian.txt', np.transpose([Pk_dumb.k, Pk_dumb.Pk]))


# save maps, A and Pk
np.save('%s/Images_M_%s_LH_z=0.00.npy'%(root_out,sim), maps_total)
np.save('%s/Power_spectra_%s.npy'%(root_out,sim),      Pk_total)
np.savetxt('%s/params_%s.txt'%(root_out,sim),          A)
