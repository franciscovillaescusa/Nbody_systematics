from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Ellipse, Rectangle
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

fig = figure(figsize=(14,16))

gs   = gridspec.GridSpec(4,6)
ax1  = plt.subplot(gs[0])
ax2  = plt.subplot(gs[1])
ax3  = plt.subplot(gs[2])
ax4  = plt.subplot(gs[3])
ax5  = plt.subplot(gs[4])
ax6  = plt.subplot(gs[5])
ax7  = plt.subplot(gs[6])
ax8  = plt.subplot(gs[7])
ax9  = plt.subplot(gs[8])
ax10 = plt.subplot(gs[9])
ax11 = plt.subplot(gs[10])
ax12 = plt.subplot(gs[11])
ax13 = plt.subplot(gs[12])
ax14 = plt.subplot(gs[13])
ax15 = plt.subplot(gs[14])
ax16 = plt.subplot(gs[15])
ax17 = plt.subplot(gs[16])
ax18 = plt.subplot(gs[17])
ax19 = plt.subplot(gs[18])
ax20 = plt.subplot(gs[19])
ax21 = plt.subplot(gs[20])
ax22 = plt.subplot(gs[21])
ax23 = plt.subplot(gs[22])
ax24 = plt.subplot(gs[23])

bar1 = axes([0.913,0.828,0.02,0.221])
bar2 = axes([0.913,0.602,0.02,0.221])

x_coord = 180
y_coord = 120
dx      = 76
dy      = 76
slice_num = 4
root  = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/same_seed'

gs.update(hspace=0.01,wspace=0.02,bottom=0.6,top=1.05)

f_out = 'comparison_slices.pdf'

# read Gadget data
f1        = '%s/Images_M_Gadget_fiducial_z=0.00.npy'%root
df_Gadget = np.load(f1)
df_Gadget = df_Gadget[slice_num]

for sim,axa,axb,axc,axd in zip(['Gadget', 'Abacus', 'CUBEP3M',
                                'PKDGrav', 'Ramses', 'Enzo7'],
                               [ax1,ax2,ax3,ax4,ax5,ax6],
                               [ax7,ax8,ax9,ax10,ax11,ax12],
                               [ax13,ax14,ax15,ax16,ax17,ax18],
                               [ax19,ax20,ax21,ax22,ax23,ax24]):
    
    f1    = '%s/Images_M_%s_fiducial_z=0.00.npy'%(root,sim)
    if sim=="Enzo6":  label = 'Enzo'
    else:             label = sim

    axa.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
    axa.yaxis.set_major_formatter( NullFormatter() )   #unset y label
    axa.axis('off')
    axb.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
    axb.yaxis.set_major_formatter( NullFormatter() )   #unset y label
    axb.axis('off')
    axc.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
    axc.yaxis.set_major_formatter( NullFormatter() )   #unset y label
    axc.axis('off')
    axd.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
    axd.yaxis.set_major_formatter( NullFormatter() )   #unset y label
    axd.axis('off')

    rect1 = Rectangle((x_coord, y_coord), dx, dy, linewidth=1, edgecolor='k',
                     facecolor='none')
    rect2 = Rectangle((x_coord, y_coord), dx, dy, linewidth=1, edgecolor='k',
                     facecolor='none')
    axa.add_patch(rect1)
    if label!='Gadget':
        axd.add_patch(rect2)
    
    df = np.load(f1)
    df = df[slice_num]
    print(df.shape)

    print('%.3e < df < %.3e'%(np.min(df), np.max(df)))
    min_density, max_density = 1e10, 8e14

    cax1 = axa.imshow(df,cmap=get_cmap('nipy_spectral'),origin='lower',
                    interpolation='bicubic',
                    #extent=[x_min, x_max, y_min, y_max],
                    #vmin=min_density,vmax=max_density)
                    norm = LogNorm(vmin=min_density,vmax=max_density))

    region = df[y_coord:y_coord+dy, x_coord:x_coord+dx]
    region_Gadget = df_Gadget[y_coord:y_coord+dy, x_coord:x_coord+dx]
    
    cax = axb.imshow(region,
                     cmap=get_cmap('nipy_spectral'),origin='lower',
                     interpolation='bicubic',
                     #extent=[x_min, x_max, y_min, y_max],
                     #vmin=min_density,vmax=max_density)
                     norm = LogNorm(vmin=min_density,vmax=max_density))

    cax = axc.imshow(region/region_Gadget,cmap=get_cmap('seismic'),origin='lower', 
                     interpolation='bicubic',
                     vmin=0.5,vmax=1.5)
    
    cax2 = axd.imshow(df/df_Gadget,cmap=get_cmap('seismic'),origin='lower', 
                      interpolation='bicubic',
                      vmin=0.5,vmax=1.5)

    
    #cbar = fig.colorbar(mappable=None, cax=cax1, ax=bar1) #in ax2 colorbar of ax1
    cbar = fig.colorbar(cax1, bar1) #in ax2 colorbar of ax1
    cbar.set_label(r"$\Sigma\,[h^{-1}M_\odot/(h^{-1}{\rm Mpc})^2]$",
                   fontsize=14,labelpad=5)
    cbar = fig.colorbar(cax2, bar2) #in ax2 colorbar of ax1
    cbar.set_label(r"$\Sigma/\Sigma_{\rm Gadget}$",
                   fontsize=14,labelpad=5)
    #cbar.ax.tick_params(labelsize=10)  #to change size of ticks

    
    #p1,=ax1.plot(x,y,linestyle='-',marker='None')


    #place a label in the plot
    #ax1.text(0.2,0.1, r"$z=4.0$", fontsize=22, color='k',transform=ax1.transAxes)

    #legend
    #ax1.legend([p1,p2],
    #           [r"$z=3$",
    #            r"$z=4$"],
    #           loc=0,prop={'size':18},ncol=1,frameon=True)
    
    #columnspacing=2,labelspacing=2)



    #ax.text(0.-1,0.4, label, fontsize=18, color='k',transform=ax.transAxes,
    #        rotation='vertical')
    axa.set_title(label,y=1.0,size=16)
    if label!='Gadget':
        axd.set_title('%s/Gadget'%label,y=-0.15,size=13)
    #title('About as simple as it gets, folks')
    #suptitle('About as simple as it gets, folks')  #for title with several panels
    #grid(True)
    #show()
savefig(f_out, bbox_inches='tight')
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
