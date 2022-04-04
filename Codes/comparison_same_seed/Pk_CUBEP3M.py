import numpy as np
import sys,os
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL

rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3

#################################### INPUT ###########################################
root    = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/Sims/CUBEP3M/comparison'
snaps   = ['%s/nc512_pp_range0.npy'%root,
           '%s/nc512_pp_range1.npy'%root,
           '%s/nc512_pp_range2.npy'%root,
           '%s/nc1024_pp_range0.npy'%root,
           '%s/nc1024_pp_range2.npy'%root,
           '%s/nc1024_pp_range2_rsoft0.5.npy'%root]

fouts   = ['Pk_CUBEP3M_nc512_pp0.txt',
           'Pk_CUBEP3M_nc512_pp1.txt',
           'Pk_CUBEP3M_nc512_pp2.txt',
           'Pk_CUBEP3M_nc1024_pp0.txt',
           'Pk_CUBEP3M_nc1024_pp2.txt',
           'Pk_CUBEP3M_nc1024_pp2_rsoft05.txt']
           
grid    = 512
MAS     = 'CIC'
axis    = 0
threads = 1
######################################################################################


for snap,fout in zip(snaps,fouts):
    BoxSize = 25.0 #Mpc/h
    pos     = np.load(snap).astype(np.float32)

    # compute density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=None, verbose=True)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    #delta = np.log10(1.0+delta)

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis, MAS=None, threads=threads, verbose=True)
    np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk[:,0]]))
