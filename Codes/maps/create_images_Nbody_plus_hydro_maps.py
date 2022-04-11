import numpy as np
import sys,os


##################################### INPUT ###########################################
root = '/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/2D_maps/data'

# maps
f_maps_hydro = '%s/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy'%root
f_maps_nbody = '%s/Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy'%root
f_maps_out   = '../../data/maps/Maps_Mtot_IllustrisTNG_Nbody+Hydro_LH_z=0.00.npy'

# parameters
f_params_hydro = '%s/params_IllustrisTNG.txt'%root
f_params_nbody = '%s/params_Nbody_IllustrisTNG.txt'%root
f_params_out   = '../../data/maps/params_IllustrisTNG_Nbody+Hydro_LH.txt'
#######################################################################################

# read the maps
maps_hydro = np.load(f_maps_hydro)
maps_nbody = np.load(f_maps_nbody)
maps       = np.vstack((maps_hydro,maps_nbody))
np.save(f_maps_out, maps)

print(maps_hydro.shape)
print(maps_nbody.shape)
print(maps.shape)

# read the parameters
params_hydro = np.loadtxt(f_params_hydro)
params_nbody = np.loadtxt(f_params_nbody)
params       = np.vstack((params_hydro,params_nbody))
np.savetxt(f_params_out, params)

print(params_hydro.shape)
print(params_nbody.shape)
print(params.shape)

print(np.where(params_hydro!=params_nbody))
