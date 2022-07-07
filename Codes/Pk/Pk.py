import numpy as np
import sys,os
import MAS_library as MASL
import Pk_library as PKL
import readgadget
import pynbody
import h5py, yt
import units_library as UL
from michaels_functions import center_and_r_vir, remove_bulk_velocity, read_unit_from_info
#from astropy.table import Table
import asdf

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


# This routine computes the power spectrum given the positions of the particles
def compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout):

    # compute density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=None, verbose=True)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    #delta = np.log10(1.0+delta)

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis, MAS=None, threads=threads, verbose=True)
    np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk[:,0]]))

#################################### INPUT ###########################################
root  = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/Sims/same_seed'

snap1 = '%s/Gadget/snap_033.hdf5'%root
fout1 = 'Pk_Gadget.txt'

snap2 = '%s/Ramses/output_00035'%root
fout2 = 'Pk_Ramses.txt'

snap3 = '%s/Ramses_HR/output_00035'%root
fout3 = 'Pk_Ramses_HR.txt'

snap4 = '%s/Abacus/z0.000.asdf'%root
fout4 = 'Pk_Abacus.txt'

snap5 = '%s/PKDGrav/run.00100'%root
fout5 = 'Pk_PKDGrav.txt'

snap6 = '%s/CUBEP3M/nc1024_pp_range2.npy'%root
fout6 = 'Pk_cube_nc1024_pp2.txt'

snap7 = '%s/PKDGrav_HR/run.00100'%root
fout7 = 'Pk_PKDGrav_HR.txt'

snap8 = '%s/Gadget_HR/snap_005.hdf5'%root
fout8 = 'Pk_Gadget_HR.txt'

snap9 = '%s/Gizmo/snap_005.hdf5'%root
fout9 = 'Pk_Gizmo.txt'

snap10 = '%s/Enzo/amr_14/RD0007/RD0007'%root
fout10 = 'Pk_Enzo4.txt'

snap11 = '%s/Abacus_HR/z0.000.asdf'%root
fout11 = 'Pk_Abacus_HR.txt'

snap12 = '%s/CUBEP3M/nc2048_pp_range2_ic_np512/0.000x.npy'%root
fout12 = 'Pk_cube_HR_nc2048_pp2.txt'

# Pk parameters
grid    = 512
MAS     = 'CIC'
axis    = 0
threads = 1
######################################################################################


############ Gadget #############
# read header
ptype    = [1] 
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

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout1)


############ Gadget_HR #############
# read header
ptype    = [1] 
header   = readgadget.header(snap8)
BoxSize  = header.boxsize/1e3  #Mpc/h
Nall     = header.nall         #Total number of particles
Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
Omega_m  = header.omega_m      #value of Omega_m
Omega_l  = header.omega_l      #value of Omega_l
h        = header.hubble       #value of h
redshift = header.redshift     #redshift of the snapshot
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

# read positions, velocities and IDs of the particles
pos = readgadget.read_block(snap8, "POS ", ptype)/1e3 #positions in Mpc/h
print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout8)


############ Gizmo #############
# read header
ptype    = [1] 
#header   = readgadget.header(snap9)
#BoxSize  = header.boxsize/1e3  #Mpc/h
#Nall     = header.nall         #Total number of particles
#Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
#Omega_m  = header.omega_m      #value of Omega_m
#Omega_l  = header.omega_l      #value of Omega_l
#h        = header.hubble       #value of h
#redshift = header.redshift     #redshift of the snapshot
#Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

# read positions, velocities and IDs of the particles
BoxSize = 25.0 #Mpc/h
f = h5py.File(snap9, 'r')
pos = f['PartType1/Coordinates'][:]/1e3
pos = pos.astype(np.float32)
print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout9)




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

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout2)


############ Ramses_HR ###########
data = pynbody.load(snap3)
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

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout3)


############ Abacus ###########
BoxSize = 25.0 #Mpc/h
af  = asdf.open(snap4)
pos = af['data']['pos'] + BoxSize/2.0
meta = af['meta']
print(pos)
#print(meta)
print(np.min(pos[:,0]), np.max(pos[:,0]))
print(np.min(pos[:,1]), np.max(pos[:,1]))
print(np.min(pos[:,2]), np.max(pos[:,2]))

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout4)


BoxSize = 25.0 #Mpc/h
af  = asdf.open(snap11)
pos = af['data']['pos'] + BoxSize/2.0
meta = af['meta']
print(pos)
#print(meta)
print(np.min(pos[:,0]), np.max(pos[:,0]))
print(np.min(pos[:,1]), np.max(pos[:,1]))
print(np.min(pos[:,2]), np.max(pos[:,2]))

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout11)


############ PKDGrav ###########
BoxSize = 25.0 #Mpc/h
npart, header, p = read_tipsy(snap5)
pos  = (p['pos']+0.5)*BoxSize  #Mpc/h
vel  = p['vel']*100.0*BoxSize*np.sqrt(3.0/(8.0*np.pi)) #km/s
mass = p['mass']*BoxSize**3*rho_crit #Msun/h
Omega_m = np.sum(mass, dtype=np.float64)/BoxSize**3/rho_crit
print('scale factor: %.3f'%(header['a']))
print('redshift:     %.3f'%(1.0/header['a']-1))
print('particles:    %d'%npart)
print('Omega_m:      %.4f'%Omega_m)
print('%.3f < X %.3f'%(np.min(pos[:,0]), np.max(pos[:,0])))
print('%.3f < Y %.3f'%(np.min(pos[:,1]), np.max(pos[:,1])))
print('%.3f < Z %.3f'%(np.min(pos[:,2]), np.max(pos[:,2])))

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout5)


############ PKDGrav_HR ###########
BoxSize = 25.0 #Mpc/h
npart, header, p = read_tipsy(snap7)
pos  = (p['pos']+0.5)*BoxSize  #Mpc/h
vel  = p['vel']*100.0*BoxSize*np.sqrt(3.0/(8.0*np.pi)) #km/s
mass = p['mass']*BoxSize**3*rho_crit #Msun/h
Omega_m = np.sum(mass, dtype=np.float64)/BoxSize**3/rho_crit
print('scale factor: %.3f'%(header['a']))
print('redshift:     %.3f'%(1.0/header['a']-1))
print('particles:    %d'%npart)
print('Omega_m:      %.4f'%Omega_m)
print('%.3f < X %.3f'%(np.min(pos[:,0]), np.max(pos[:,0])))
print('%.3f < Y %.3f'%(np.min(pos[:,1]), np.max(pos[:,1])))
print('%.3f < Z %.3f'%(np.min(pos[:,2]), np.max(pos[:,2])))

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout7)


########### CUBEP3M #############
BoxSize = 25.0 #Mpc/h
pos     = np.load(snap6).astype(np.float32)
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout6)

BoxSize = 25.0 #Mpc/h
pos     = np.load(snap12).astype(np.float32)
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout12)


########### Enzo #############
# read the positions of the particles
data    = yt.load(snap10)
ad      = data.all_data()
BoxSize = data.domain_width.in_units('Mpccm/h').d[0]              #Mpc/h
pos     = ad[('nbody','particle_position')].in_units('Mpccm/h').d #Mpc/h
pos     = pos.astype(np.float32)

# compute Pk
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout10)

