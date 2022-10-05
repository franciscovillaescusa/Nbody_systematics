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

x_min, x_max = 0.3, 100
y_min, y_max = 7e-4, 1e3


fig=figure(figsize=(10/1.3,5/1.3))
ax1=fig.add_subplot(111) 

ax1.set_xscale('log')
ax1.set_xlim([x_min,x_max])
ax1.set_ylim([0.93,1.03])

ax1.set_xlabel(r'$k\/[h\/{\rm Mpc}^{-1}]$',fontsize=18)
ax1.set_ylabel(r'$P(k)/P_{\rm Gadget}(k)$',fontsize=18)


f_out='Pk_ratio.pdf'

root = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/PUBLIC/Results/Pk'
f1   = '%s/Pk_Gadget.txt'%root
f2   = '%s/Pk_Gadget_HR.txt'%root
f3   = '%s/Pk_PKDGrav.txt'%root
f4   = '%s/Pk_PKDGrav_HR.txt'%root
f5   = '%s/Pk_Abacus.txt'%root
f6   = '%s/Pk_Ramses.txt'%root
f7   = '%s/Pk_Ramses_HR.txt'%root
f8   = '%s/Pk_CUBEP3M.txt'%root
f9   = '%s/Pk_Gizmo.txt'%root
f10  = '%s/Pk_Enzo5.txt'%root

data1  = np.loadtxt(f1) 
data2  = np.loadtxt(f2) 
data3  = np.loadtxt(f3) 
data4  = np.loadtxt(f4) 
data5  = np.loadtxt(f5) 
data6  = np.loadtxt(f6) 
data7  = np.loadtxt(f7) 
data8  = np.loadtxt(f8) 
data9  = np.loadtxt(f9) 
data10 = np.loadtxt(f10) 

ax1.fill_between([x_min,x_max],[1.01,1.01],[0.99,0.99],color='k',alpha=0.3)
ax1.fill_between([x_min,x_max],[1.02,1.02],[0.98,0.98],color='k',alpha=0.1)

# Gadget
p1,=ax1.plot(data2[:,0],data1[:,1]/data1[:,1],linestyle='-',marker='None',c='b')
#p2,=ax1.plot(data2[:,0],data2[:,1]/data1[:,1],linestyle='-',marker='None',c='darkblue')

# PKDGrav
p3,=ax1.plot(data2[:,0],data3[:,1]/data1[:,1],linestyle='-',marker='None',c='r')
#p4,=ax1.plot(data2[:,0],data4[:,1]/data1[:,1],linestyle='-',marker='None',c='darkred')

# Abacus
p5,=ax1.plot(data2[:,0],data5[:,1]/data1[:,1],linestyle='-',marker='None',c='purple')

# Ramses
p6,=ax1.plot(data2[:,0],data6[:,1]/data1[:,1],linestyle='-',marker='None',c='green')
#p7,=ax1.plot(data2[:,0],data7[:,1]/data1[:,1],linestyle='-',marker='None',c='darkgreen')

# CUBEP3M
p8,=ax1.plot(data2[:,0],data8[:,1]/data1[:,1],linestyle='-',marker='None',c='orange')

# Gizmo
#p9,=ax1.plot(data2[:,0],data9[:,1]/data1[:,1],linestyle='-',marker='None',c='cyan')

# Enzo
p10,=ax1.plot(data2[:,0],data10[:,1]/data1[:,1],linestyle='-',marker='None',c='magenta')

ax1.plot([32,32],[0.9,1.1],linestyle='--',marker='None',c='k')



#place a label in the plot
ax1.text(0.05,0.9, r"$z=0$", fontsize=18, color='k',transform=ax1.transAxes)

#legend
ax1.legend([p1,p3,p5,p6,p8,p10],
           ["Gadget", "PKDGrav",
            "Abacus", "Ramses", 
            "CUBEP3M", "Enzo"],
           loc=3,prop={'size':10},ncol=3,frameon=True)
"""
ax1.legend([p1,p8, p2,p9, p3, p6, p4,p5,p7,p10],
           ["Gadget", "Gadget HR", 
            "PKDGrav", "PKDGrav HR", 
            "Abacus", "CUBEP3M", 
            "Ramses", "Ramses HR", 
            "Gizmo", "Enzo"],
           loc=0,prop={'size':10},ncol=5,frameon=True, bbox_to_anchor=(0.87, 1.03))
"""            
  




#ax1.set_title(r'$\sum m_\nu=0.0\/{\rm eV}$',position=(0.5,1.02),size=18)
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
