import numpy as np
import sys,os
import MAS_library as MASL
import Pk_library as PKL
import readgadget
import pynbody
from michaels_functions import center_and_r_vir, remove_bulk_velocity, read_unit_from_info
#from astropy.table import Table
import asdf

#################################### INPUT ###########################################
snap1   = '/mnt/ceph/users/camels/Sims/Ramses_DM/CV_G_0/ICs/ics'
snap2   = '/mnt/ceph/users/camels/Sims/Ramses/dmo/output_00001'
grid    = 512
MAS     = 'CIC'
axis    = 0
threads = 1
######################################################################################


############ Gadget #############
ptype = [1] 

# read header
header   = readgadget.header(snap1)
BoxSize  = header.boxsize/1e3  #Mpc/h
Nall     = header.nall         #Total number of particles
Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
Omega_m  = header.omega_m      #value of Omega_m
Omega_l  = header.omega_l      #value of Omega_l
h        = header.hubble       #value of h
redshift = header.redshift     #redshift of the snapshot
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

# read positions, velocities and IDs of the particles
pos = readgadget.read_block(snap1, "POS ", ptype)/1e3 #positions in Mpc/h

print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))

# compute density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)
MASL.MA(pos, delta, BoxSize, MAS, W=None, verbose=True)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

# compute power spectrum
Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads=threads, verbose=True)
np.savetxt('Pk_ICs_Gadget.txt', np.transpose([Pk.k3D, Pk.Pk[:,0]]))


############ Ramses ###########
data = pynbody.load(snap2)
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
print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))

# compute density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)
MASL.MA(pos, delta, BoxSize, MAS, W=None, verbose=True)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

# compute power spectrum
Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads=threads, verbose=True)
np.savetxt('Pk_ICs_Ramses.txt', np.transpose([Pk.k3D, Pk.Pk[:,0]]))


