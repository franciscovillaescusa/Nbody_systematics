import numpy as np
import sys,os
import MAS_library as MASL
import Pk_library as PKL
import readgadget
import pynbody
import h5py, yt
import units_library as UL
from michaels_functions import remove_bulk_velocity, read_unit_from_info
#from astropy.table import Table
import asdf

rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3

# routine to read PKDGrav snapshots
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
def compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log):

    # compute density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=None, verbose=True)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    if log:  delta = np.log10(2.0 + delta)

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis, MAS=MAS, threads=threads, verbose=True)
    np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk[:,0]]))

#################################### INPUT ###########################################
root     = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/Sims/same_seed'
root_out = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/PUBLIC/Results/Pk'
log      = True #whether compute standard Pk(False) or Pk of log of field(True)

if log:  suffix = 'log_'
else:    suffix = ''

# Gadget
snap1 = '%s/Gadget/snap_033.hdf5'%root
snap2 = '%s/Gadget_HR/snap_005.hdf5'%root
fout1 = '%s/Pk_%sGadget.txt'%(root_out,suffix)
fout2 = '%s/Pk_%sGadget_HR.txt'%(root_out,suffix)

# Ramses
snap3 = '%s/Ramses/output_00035'%root
snap4 = '%s/Ramses_HR/output_00035'%root
fout3 = '%s/Pk_%sRamses.txt'%(root_out,suffix)
fout4 = '%s/Pk_%sRamses_HR.txt'%(root_out,suffix)

# Abacus
snap5 = '%s/Abacus/z0.000.asdf'%root
snap6 = '%s/Abacus_HR/z0.000.asdf'%root
fout5 = '%s/Pk_%sAbacus.txt'%(root_out,suffix)
fout6 = '%s/Pk_%sAbacus_HR.txt'%(root_out,suffix)

# PKDGrav
snap7 = '%s/PKDGrav/run.00100'%root
snap8 = '%s/PKDGrav_HR/run.00100'%root
fout7 = '%s/Pk_%sPKDGrav.txt'%(root_out,suffix)
fout8 = '%s/Pk_%sPKDGrav_HR.txt'%(root_out,suffix)

# CUBEP3M
snap9  = '%s/CUBEP3M/nc1024_pp_range2.npy'%root
snap10 = '%s/CUBEP3M/nc2048_pp_range2_ic_np512/0.000x.npy'%root
fout9  = '%s/Pk_%sCUBEP3M.txt'%(root_out,suffix)
fout10 = '%s/Pk_%sCUBEP3M_HR.txt'%(root_out,suffix)

# Enzo
#snap11 = '%s/Enzo/default/RD0003/RD0003'%root
#fout11 = '%s/Pk_%sEnzo1.txt'%(root_out,suffix)
#snap11 = '%s/Enzo/small_courant/RD0003/RD0003'%root
#fout11 = '%s/Pk_%sEnzo2.txt'%(root_out,suffix)
#snap11 = '%s/Enzo/amr/RD0003/RD0003'%root
#fout11 = '%s/Pk_%sEnzo3.txt'%(root_out,suffix)
#snap11 = '%s/Enzo/amr_14/RD0007/RD0007'%root
#fout11 = '%s/Pk_%sEnzo4.txt'%(root_out,suffix)
snap11 = '%s/Enzo/amr_14_smallc/RD0007/RD0007'%root
fout11 = '%s/Pk_%sEnzo5.txt'%(root_out,suffix)

# Gizmo
snap12 = '%s/Gizmo/snap_005.hdf5'%root
fout12 = '%s/Pk_%sGizmo.txt'%(root_out,suffix)

# Pk parameters
grid    = 512
MAS     = 'CIC'
axis    = 0
threads = 1
######################################################################################

############ Gadget #############
for snap,fout in zip([snap1,snap2], [fout1,fout2]):
    header  = readgadget.header(snap)
    BoxSize = header.boxsize/1e3  #Mpc/h
    pos = readgadget.read_block(snap, "POS ", [1])/1e3 #positions in Mpc/h
    print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
    print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
    print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))
    compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log)
#################################

############# Ramses ############
for snap,fout in zip([snap3,snap4],[fout3,fout4]):

    # read header
    data = pynbody.load(snap)
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
    compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log)
#################################

############# Abacus ############
for snap,fout in zip([snap5,snap6],[fout5,fout6]):
    BoxSize = 25.0 #Mpc/h
    af      = asdf.open(snap)
    pos     = af['data']['pos'] + BoxSize/2.0
    meta    = af['meta']
    #print(meta)
    print(np.min(pos[:,0]), np.max(pos[:,0]))
    print(np.min(pos[:,1]), np.max(pos[:,1]))
    print(np.min(pos[:,2]), np.max(pos[:,2]))
    compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log)
#################################

############# PKDGrav ###########
for snap,fout in zip([snap7,snap8],[fout7,fout8]):
    BoxSize = 25.0 #Mpc/h
    npart, header, p = read_tipsy(snap)
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
    compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log)
#################################

########### CUBEP3M #############
for snap,fout in zip([snap9,snap10],[fout9,fout10]):
    BoxSize = 25.0 #Mpc/h
    pos     = np.load(snap).astype(np.float32)
    compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log)
#################################

########### Enzo #############
for snap,fout in zip([snap11],[fout11]):
    data    = yt.load(snap)
    ad      = data.all_data()
    BoxSize = data.domain_width.in_units('Mpccm/h').d[0]              #Mpc/h
    pos     = ad[('nbody','particle_position')].in_units('Mpccm/h').d #Mpc/h
    pos     = pos.astype(np.float32)
    compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout, log)
#################################

############# Gizmo #############
BoxSize = 25.0 #Mpc/h
f       = h5py.File(snap12, 'r')
pos     = f['PartType1/Coordinates'][:]/1e3
pos     = pos.astype(np.float32)
print('%.3f < X < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
print('%.3f < Y < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
print('%.3f < Z < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))
compute_Pk(grid, pos, BoxSize, MAS, axis, threads, fout12, log)
#################################


