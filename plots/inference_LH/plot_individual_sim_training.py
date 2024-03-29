from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

############################### figure ###########################
#fig=figure(figsize=(15,10))     #give dimensions to the figure
##################################################################

################################ INPUT #######################################
#axes range
##############################################################################

############################ subplots ############################
#gs = gridspec.GridSpec(2,1,height_ratios=[5,2])
#ax1=plt.subplot(gs[0])
#ax2=plt.subplot(gs[1])

#make a subplot at a given position and with some given dimensions
#ax2=axes([0.4,0.55,0.25,0.1])

#gs.update(hspace=0.0,wspace=0.4,bottom=0.6,top=1.05)
#subplots_adjust(left=None, bottom=None, right=None, top=None,
#                wspace=0.5, hspace=0.5)

#set minor ticks
#ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
#ax1.yaxis.set_minor_locator(AutoMinorLocator(4))


#ax1.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
#ax1.yaxis.set_major_formatter( NullFormatter() )   #unset y label

# custom xticks 
#ax1.set_xticks([0.25, 0.5, 1.0])
#ax1.set_yticks([0.25, 0.5, 1.0])
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) #for log 


#ax1.get_yaxis().set_label_coords(-0.2,0.5)  #align y-axis for multiple plots
##################################################################

##################### special behaviour stuff ####################
#to show error missing error bars in log scale
#ax1.set_yscale('log',nonposy='clip')  #set log scale for the y-axis

#set the x-axis in %f format instead of %e
#ax1.xaxis.set_major_formatter(ScalarFormatter()) 

#set size of ticks
#ax1.tick_params(axis='both', which='major', labelsize=10)
#ax1.tick_params(axis='both', which='minor', labelsize=8)

#set the position of the ylabel 
#ax1.yaxis.set_label_coords(-0.2, 0.4)

#set yticks in scientific notation
#ax1.ticklabel_format(axis='y',style='sci',scilimits=(1,4))

#set the x-axis in %f format instead of %e
#formatter = matplotlib.ticker.FormatStrFormatter('$%.2e$') 
#ax1.yaxis.set_major_formatter(formatter) 

#add two legends in the same plot
#ax5 = ax1.twinx()
#ax5.yaxis.set_major_formatter( NullFormatter() )   #unset y label 
#ax5.legend([p1,p2],['0.0 eV','0.3 eV'],loc=3,prop={'size':14},ncol=1)

#set points to show in the yaxis
#ax1.set_yticks([0,1,2])

#highlight a zoomed region
#mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none",edgecolor='purple')
##################################################################

############################ plot type ###########################
#standard plot
#p1,=ax1.plot(x,y,linestyle='-',marker='None')

#error bar plot with the minimum and maximum values of the error bar interval
#p1=ax1.errorbar(r,xi,yerr=[delta_xi_min,delta_xi_max],lw=1,fmt='o',ms=2,
#               elinewidth=1,capsize=5,linestyle='-') 

#filled area
#p1=ax1.fill_between([x_min,x_max],[1.02,1.02],[0.98,0.98],color='k',alpha=0.2)

#hatch area
#ax1.fill([x_min,x_min,x_max,x_max],[y_min,3.0,3.0,y_min],#color='k',
#         hatch='X',fill=False,alpha=0.5)

#scatter plot
#p1=ax1.scatter(k1,Pk1,c='b',edgecolor='none',s=8,marker='*')

#plot with markers
#pl4,=ax1.plot(ke3,Pk3/Pke3,marker='.',markevery=2,c='r',linestyle='None')

#set size of dashed lines
#ax.plot([0, 1], [0, 1], linestyle='--', dashes=(5, 1)) #length of 5, space of 1

#image plot
#cax = ax1.imshow(densities,cmap=get_cmap('jet'),origin='lower',
#           extent=[x_min, x_max, y_min, y_max],
#           #vmin=min_density,vmax=max_density)
#           norm = LogNorm(vmin=min_density,vmax=max_density))
#cbar = fig.colorbar(cax, ax2, ax=ax1, ticks=[-1, 0, 1]) #in ax2 colorbar of ax1
#cbar.set_label(r"$M_{\rm CSF}\/[h^{-1}M_\odot]$",fontsize=14,labelpad=-50)
#cbar.ax.tick_params(labelsize=10)  #to change size of ticks

#make a polygon
#polygon = Rectangle((0.4,50.0), 20.0, 20.0, edgecolor='purple',lw=0.5,
#                    fill=False)
#ax1.add_artist(polygon)
####################################################################

########################## INPUT ########################
minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
splits = 15
#########################################################

# get the name of the files
#root = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/PUBLIC/Codes/train_test_models'
root = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/PUBLIC/Results/networks/LH'

f_ins  = ['%s/Trained_Gadget_smoothing_0_tested_Gadget_smoothing_0_z=0.00.txt'%root]
f_outs = ['Trained_Gadget_smoothing_0_tested_Gadget_smoothing_0_z=0.00.pdf']
labels = ['Train in Gadget ; test in Gadget']


"""
f_ins = ['%s/Trained_Gadget_tested_Gadget_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_Abacus_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_Ramses_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_CUBEP3M_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_PKDGrav_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_Gadget_test_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_Gadget_NCV_z=0.00.txt'%root,
         '%s/Trained_Gadget_tested_Gadget_NCV2_z=0.00.txt'%root,
         '%s/Trained_Gadget_multiresolution_tested_Gadget_multiresolution_z=0.00.txt'%root,
         '%s/Trained_Gadget_multiresolution_tested_PKDGrav_z=0.00.txt'%root,
         '%s/Trained_Gadget_multiresolution_tested_Gadget_z=0.00.txt'%root,
         '%s/Trained_Gadget_multiresolution_tested_Ramses_z=0.00.txt'%root,
         '%s/Trained_Gadget_multiresolution_tested_COLA_z=0.00.txt'%root,
         '%s/Trained_Gadget_multiresolution_tested_Gadget_test_z=0.00.txt'%root,
         '%s/Trained_Gadget_Nbody+Hydro_tested_Gadget_z=0.00.txt'%root,
         '%s/Trained_Gadget_Nbody+Hydro_tested_Abacus_z=0.00.txt'%root,
         '%s/Trained_Gadget_Nbody+Hydro_tested_Ramses_z=0.00.txt'%root]

f_outs = ['Trained_Gadget_tested_Gadget_z=0.00.pdf',
          'Trained_Gadget_tested_Abacus_z=0.00.pdf',
          'Trained_Gadget_tested_Ramses_z=0.00.pdf',
          'Trained_Gadget_tested_CUBEP3M_z=0.00.pdf',
          'Trained_Gadget_tested_PKDGrav_z=0.00.pdf',
          'Trained_Gadget_tested_Gadget_test_z=0.00.pdf',
          'Trained_Gadget_tested_Gadget_NCV_z=0.00.pdf',
          'Trained_Gadget_tested_Gadget_NCV2_z=0.00.pdf',
          'Trained_Gadget_multiresolution_tested_Gadget_multiresolution_z=0.00.pdf',
          'Trained_Gadget_multiresolution_tested_PKDGrav_z=0.00.pdf',
          'Trained_Gadget_multiresolution_tested_Gadget_z=0.00.pdf',
          'Trained_Gadget_multiresolution_tested_Ramses_z=0.00.pdf',
          'Trained_Gadget_multiresolution_tested_COLA_z=0.00.pdf',
          'Trained_Gadget_multiresolution_tested_Gadget_test_z=0.00.pdf',
          'Trained_Gadget_Nbody+Hydro_tested_Gadget_z=0.00.pdf',
          'Trained_Gadget_Nbody+Hydro_tested_Abacus_z=0.00.pdf',
          'Trained_Gadget_Nbody+Hydro_tested_Ramses_z=0.00.pdf']

labels = ['Train on Gadget ----> test on Gadget',
          'Train on Gadget ----> test on Abacus',
          'Train on Gadget ----> test on Ramses',
          'Train on Gadget ----> test on CUBEP3M',
          'Train on Gadget ----> test on PKDGrav',
          'Train on Gadget ----> test on Gadget test',
          'Train on Gadget ----> test on Gadget NCV',
          'Train on Gadget ----> test on Gadget NCV2',
          'Train on Gadget MR ----> test on Gadget MR',
          'Train on Gadget MR ----> test on PKDGrav',
          'Train on Gadget MR ----> test on Gadget',
          'Train on Gadget MR ----> test on Ramses',
          'Train on Gadget MR ----> test on COLA',
          'Train on Gadget MR ----> test on Gadget test',
          'Train on Gadget Nbody+Hydro ----> test on Gadget',
          'Train on Gadget Nbody+Hydro ----> test on Abacus',
          'Train on Gadget Nbody+Hydro ----> test on Ramses']
"""

for fin, fout, label in zip(f_ins, f_outs, labels):
    np.random.seed(17) #4

    # read data
    data = np.loadtxt(fin)

    # compute statistics
    #abs_error = np.zeros(2, dtype=np.float32)
    #bias      = np.zeros(2, dtype=np.float32)
    rel_error = np.zeros(2, dtype=np.float32)
    R2        = np.zeros(2, dtype=np.float32)
    RMSE      = np.zeros(2, dtype=np.float32)
    chi2      = np.zeros(2, dtype=np.float32)

    for i in range(2):
        true, mean, error = data[:,i], data[:,6+i], data[:,12+i]
        rel_error[i] = np.mean(np.absolute(true-mean)/true)
        R2[i]        = 1.0 - np.mean((true-mean)**2)/np.mean((true-np.mean(true))**2)
        RMSE[i]      = np.sqrt(np.mean((true-mean)**2))
        chi2[i]      = np.mean((true-mean)**2/error**2)
        #abs_error[i] = np.mean(np.absolute(data[:,12+i]))
        #rel_error[i] = np.mean(np.absolute(data[:,12+i])/np.absolute(data[:,6+i]))
        #bias[i]      = np.mean(data[:,6+i] - data[:,i])

    num_sims     = data.shape[0]/8/splits
    unique_maps  = num_sims*splits
    maps_per_sim = 15*8
    unique_indexes = np.arange(num_sims, dtype=np.int32)*maps_per_sim #add some offset here

    fig = figure(figsize=(16,6))
    ax1 = fig.add_subplot(121) 
    ax2 = fig.add_subplot(122) 

    subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.10, hspace=0.1)

    for ax in [ax1,ax2]:
        ax.set_xlabel(r'${\rm True}$',      fontsize=18)
    for ax in [ax1]:
        ax.set_ylabel(r'${\rm Inference}$',fontsize=18)

    Om = data[:,0]
    indexes = np.argsort(Om)
    data = data[indexes]

    indexes = np.random.choice(unique_indexes,50,replace=False)
    Om = data[indexes,0]

    norm = matplotlib.colors.Normalize(vmin=min(Om), vmax=max(Om), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='brg')
    time_color = np.array([(mapper.to_rgba(v)) for v in Om])

    for i,ax,c,minimum,maximum in zip([0,1],[ax1,ax2],['r','b'], [0.1, 0.6], [0.5, 1.0]):

        # read true, prediction and error
        T, P, E = data[:,i], data[:,6+i], data[:,12+i]

        for k,j in enumerate(indexes):

            ax.errorbar(T[j], P[j], yerr=E[j], lw=1, fmt='o', ms=2,
                        elinewidth=1, capsize=0, linestyle='None', c=time_color[k]) 

        ax.plot([minimum,maximum],[minimum,maximum], ls='-', c='k')
        ax.text(0.73,0.22, r"$\epsilon=%.2f$"%(100*rel_error[i])+'%',
                fontsize=16, color='k', transform=ax.transAxes)
        ax.text(0.73,0.16, r"${\rm RMSE}=%.4f$"%RMSE[i],
                fontsize=16, color='k', transform=ax.transAxes)
        ax.text(0.73,0.10, r"$R^2=%.3f$"%R2[i], fontsize=16, color='k',
                transform=ax.transAxes)
        ax.text(0.73,0.04, r"$\chi^2=%.2f$"%chi2[i], fontsize=16, color='k',
                transform=ax.transAxes)

    #place a label in the plot
    ax1.text(0.05,0.87, r"$\Omega_{\rm m}$", fontsize=20, color='k',
             transform=ax1.transAxes)
    ax2.text(0.05,0.87, r"$\sigma_8$",       fontsize=20, color='k',
             transform=ax2.transAxes)

    #ax1.set_title(r'$\sum m_\nu=0.0\/{\rm eV}$',position=(0.5,1.02),size=18)
    #title('About as simple as it gets, folks')
    suptitle(label, size=20, position=(0.5,0.94))  #for title with several panels
    #grid(True)
    #show()
    savefig(fout, bbox_inches='tight')
    close(fig)










###############################################################################
#some useful colors:

#'darkseagreen'
#'yellow'
#"hotpink"
#"gold
#"fuchsia"
#"lime"
#"brown"
#"silver"
#"cyan"
#"dodgerblue"
#"darkviolet"
#"magenta"
#"deepskyblue"
#"orchid"
#"aqua"
#"darkorange"
#"coral"
#"lightgreen"
#"salmon"
#"bisque"
