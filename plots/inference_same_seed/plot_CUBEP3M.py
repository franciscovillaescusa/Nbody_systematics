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

x_min, x_max = -2, 128
y_min, y_max = 0.05, 0.45


fig=figure(figsize=(15,5))
ax1=fig.add_subplot(211) 
ax2=fig.add_subplot(212) 

subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.0, hspace=0.02)

#ax1.set_xscale('log')
#ax1.set_yscale('log')

for ax in [ax1,ax2]:
    ax.set_xlim([x_min,x_max])
    #ax.set_ylim([y_min,y_max])
    ax.set_xticks([])

#ax1.set_xlabel(r'$k\/[h\/{\rm Mpc}^{-1}]$',fontsize=18)
ax1.set_ylabel(r'$\Omega_{\rm m}$',fontsize=18)
ax2.set_ylabel(r'$\sigma_8$',fontsize=18)

#ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
#ax1.xaxis.set_major_formatter( NullFormatter() )   #unset x label 


f_out='Inference_CUBEP3M.pdf'

ax1.plot([-10,200],[0.3175,0.3175], lw=1, c='k')
ax2.plot([-10,200],[0.834, 0.834],  lw=1, c='k')

root = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/PUBLIC/Codes/train_test_models'

f1 = '%s/Trained_Gadget_tested_CUBEP3M_nc512_pp0_z=0.00.txt'%root
f2 = '%s/Trained_Gadget_tested_CUBEP3M_nc512_pp1_z=0.00.txt'%root
f3 = '%s/Trained_Gadget_tested_CUBEP3M_nc512_pp2_z=0.00.txt'%root
f4 = '%s/Trained_Gadget_tested_CUBEP3M_nc1024_pp0_z=0.00.txt'%root
f5 = '%s/Trained_Gadget_tested_CUBEP3M_nc1024_pp2_z=0.00.txt'%root
f6 = '%s/Trained_Gadget_tested_CUBEP3M_nc1024_pp3_z=0.00.txt'%root
f7 = '%s/Trained_Gadget_tested_CUBEP3M_nc1024_pp2_rsoft0.5_z=0.00.txt'%root
f8 = '%s/Trained_Gadget_tested_CUBEP3M_nc2048_pp2_z=0.00.txt'%root

data1 = np.loadtxt(f1) 
data2 = np.loadtxt(f2) 
data3 = np.loadtxt(f3) 
data4 = np.loadtxt(f4) 
data5 = np.loadtxt(f5) 
data6 = np.loadtxt(f6) 
data7 = np.loadtxt(f7) 
data8 = np.loadtxt(f8) 

elems = 15
x = np.arange(elems)
indexes = x*8

p1=ax1.errorbar(x+elems*0+0,data1[indexes,6],yerr=data1[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None', c='b') 
p2=ax1.errorbar(x+elems*1+1,data2[indexes,6],yerr=data2[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None', c='r') 
p3=ax1.errorbar(x+elems*2+2,data3[indexes,6],yerr=data3[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None',c='g') 
p4=ax1.errorbar(x+elems*3+3,data4[indexes,6],yerr=data4[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None', c='purple') 
p5=ax1.errorbar(x+elems*4+4,data5[indexes,6],yerr=data5[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None', c='brown')
p6=ax1.errorbar(x+elems*5+5,data6[indexes,6],yerr=data6[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None',c='fuchsia') 
p7=ax1.errorbar(x+elems*6+6,data7[indexes,6],yerr=data7[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None',c='gold') 
p8=ax1.errorbar(x+elems*7+7,data8[indexes,6],yerr=data8[indexes,12],lw=1,fmt='o',ms=2,
                elinewidth=1,capsize=5,linestyle='None',c='k') 

ax2.errorbar(x+elems*0+0,data1[indexes,7],yerr=data1[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None', c='b') 
ax2.errorbar(x+elems*1+1,data2[indexes,7],yerr=data2[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None', c='r') 
ax2.errorbar(x+elems*2+2,data3[indexes,7],yerr=data3[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None',c='g') 
ax2.errorbar(x+elems*3+3,data4[indexes,7],yerr=data4[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None', c='purple') 
ax2.errorbar(x+elems*4+4,data5[indexes,7],yerr=data5[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None', c='brown')
ax2.errorbar(x+elems*5+5,data6[indexes,7],yerr=data6[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None',c='fuchsia')
ax2.errorbar(x+elems*6+6,data7[indexes,7],yerr=data7[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None',c='gold') 
ax2.errorbar(x+elems*7+7,data8[indexes,7],yerr=data8[indexes,13],lw=1,fmt='o',ms=2,
             elinewidth=1,capsize=5,linestyle='None',c='k')  





#place a label in the plot
#ax1.text(0.2,0.1, r"$z=4.0$", fontsize=22, color='k',transform=ax1.transAxes)

#legend
ax1.legend([p1,p2,p3,p4,p5,p6,p7,p8],
           [r"${\rm nc512\,\,pp0}$",
            r"${\rm nc512\,\,pp1}$",
            r"${\rm nc512\,\,pp2}$",
            r"${\rm nc1024\,\,pp0}$",
            r"${\rm nc1024\,\,pp2}$",
            r"${\rm nc1024\,\,pp3}$",
            r"${\rm nc1024\,\,pp2\,\,rsoft0.5}$",
            r"${\rm nc2048\,\,pp2}$"],
           loc=0,prop={'size':14},ncol=4,frameon=True, bbox_to_anchor=(0.2, 0.97))
            
            #columnspacing=2,labelspacing=2)




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