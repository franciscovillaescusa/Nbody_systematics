import numpy as np
import sys,os

##################################### INPUT #########################################
root         = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data'
sim          = 'COLA'
realizations = 1000
splits       = 15
redshift     = 0.0
pixels       = 256
area         = 25.0**2/256**2
#####################################################################################

# define the arrays containing all maps and the parameters
maps   = np.zeros((splits*realizations,pixels,pixels), dtype=np.float32)
params = np.zeros((realizations,6),                    dtype=np.float32) 

# do a loop over all realizations
count, i = 0, 0 
while(count<realizations):

    # get the name of the files
    f_maps   = '%s/maps/maps_%s/Images_M_%s_LH_%d_z=%.2f.npy'%(root,sim,sim,i,redshift)
    f_params = '%s/Sims/%s/%d/Params.txt'%(root,sim,i)
    if not(os.path.exists(f_maps)):
        i+=1
        continue

    # read the maps and allocate them into the big array
    data = np.load(f_maps)
    maps[count*splits:(count+1)*splits,:,:] = data

    # read the params and put them in the big array
    data = np.loadtxt(f_params)
    params[count,[0,1]]     = data[:-1]
    params[count,[2,3,4,5]] = [1,1,1,1]
    print(i,count)
    count += 1
    i += 1


# get the name of the output files and save them
fout_maps   = '%s/maps/maps_%s/Images_M_%s_LH_z=%.2f.npy'%(root,sim,sim,redshift)
fout_params = '%s/maps/maps_%s/params_%s.txt'%(root,sim,sim)
np.save(fout_maps, maps/area)
np.savetxt(fout_params, params)
