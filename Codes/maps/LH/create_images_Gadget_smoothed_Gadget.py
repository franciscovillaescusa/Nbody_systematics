import numpy as np
import scipy.ndimage
import sys,os

# This routine takes a set of maps and smooth them with a Gaussian kernel
def smooth_maps(maps, smoothing, verbose=True):
    
    if verbose:  print('Smoothing images with smoothing length: %d'%smoothing)

    # do a loop over all maps
    for i in range(maps.shape[0]):
        image = maps[i]
        maps[i] = scipy.ndimage.gaussian_filter(image, smoothing, mode='wrap')

    return maps

##################################### INPUT ###########################################
root      = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps'
smoothing = 1

# maps
f_maps_nbody = '%s/maps_Gadget/Images_M_Gadget_LH_z=0.00.npy'%root
f_maps_out   = '%s/maps_Gadget_smoothed1_Gadget/Images_M_Gadget_smoothed1_Gadget_LH_z=0.00.npy'%root

# parameters
f_params_nbody = '%s/maps_Gadget/params_Gadget.txt'%root
f_params_out   = '%s/maps_Gadget_smoothed1_Gadget/params_Gadget_smoothed1_Gadget.txt'%root
#######################################################################################

# read the maps
maps_nbody = np.load(f_maps_nbody)

# smooth the maps
maps_smoothed = np.copy(maps_nbody)
maps_smoothed = smooth_maps(maps_smoothed, smoothing, verbose=True)

# join the maps
maps = np.vstack((maps_nbody, maps_smoothed))
np.save(f_maps_out, maps)

print(maps_nbody.shape)
print(maps_smoothed.shape)
print(maps.shape)

# read the parameters
params_nbody = np.loadtxt(f_params_nbody)
params       = np.vstack((params_nbody,params_nbody))
np.savetxt(f_params_out, params)

print(params_nbody.shape)
print(params_nbody.shape)
print(params.shape)
