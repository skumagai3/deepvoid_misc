import time
import numpy as np
#import indratools as indra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, Normalize
'''
The purpose of this script is to store functions for plotting various Indra quantities in a cookbook
notebook. This script assumes the actual physical quantities have already been computed and stored as 
standard names.

Variable names of physical quantities:
pos - particle positions
vel - particle velocities
ids - particle IDs (must be sorted if you are interested in ORIGAMI)
tags - particle ORIGAMI labels
d - density contrast \delta.
p - gravitational potential \Phi.
dtfe_vel_x(y,z) - velocity fields for the x,y,z directions.
H_eig_signs - indicates signs of the eigenvalues of the Hessian of the potential. 0 means mixed sign,
1 means (+ + +) eigs (local minimum), 2 means (- - -) eigs (local maximum).

For each plotting function, there will be a height h, bottom b, and top t.
h indicates which slice (idx) to take in the YZ plane.
b indicates the lower bound of the window.
t indicates the upper bound of the window.
ALL ARE IN Mpc!!!!
'''
#---------------------------------------------------------
# Regularization options: minmax and standardize
#---------------------------------------------------------
def minmax(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))
def standardize(a):
  return (a-np.mean(a))/(np.std(a))
#---------------------------------------------------------
# Summary statistics for a Numpy array
#---------------------------------------------------------
def summary(array):
  print('### Summary Statistics ###')
  print('Shape: ',str(array.shape))
  print('Mean: ',np.mean(array))
  print('Median: ',np.median(array))
  print('Maximum: ',np.max(array))
  print('Minimum: ',np.min(array))
  print('Std deviation: ',np.std(array))
  print('Variance: ',np.var(array))
def Mpc_to_cou(x,Nm,boxsize=205.):
    # convert Mpc to mesh cell units
    d = np.round(x * (Nm/boxsize))
    return d.astype(int)
def cou_to_Mpc(x,Nm,boxsize=205.):
    # convert mesh cell units to Mpc
    return x * (boxsize/Nm)
def set_window(b,t,Nm,ax,boxsize=205.,nticks=7,fix=True,Latex=False,return_ints=True):
    '''
    Set window function for plotting. Fixes axes labels, ticks, and limits.
    This is to address the mismatch in physical and mesh cell units.
    b - lower bound in Mpc
    t - upper bound in Mpc
    Nm - number of mesh cells in one dimension
    ax - axis to plot onto
    fix - whether or not to fix the window to the nearest integer
    Latex - whether or not to use Latex in labels
    return_ints - whether or not to return the window in integers
    '''
    b_cou = Mpc_to_cou(b,Nm,boxsize)
    t_cou = Mpc_to_cou(t,Nm,boxsize)
    if fix:
        t_cou -= 1
    ax.set_xlim(b_cou,t_cou)
    ax.set_ylim(b_cou,t_cou)
    if Latex:
        ax.set_xlabel(r'$h^{-1}$ Mpc')
        ax.set_ylabel(r'$h^{-1}$ Mpc')
    else:
        ax.set_xlabel('Mpc/h')
        ax.set_ylabel('Mpc/h')
    ax.set_xticks(np.linspace(b_cou,t_cou,nticks))
    ax.set_yticks(np.linspace(b_cou,t_cou,nticks))
    if return_ints:
        ax.set_xticklabels(np.round(np.linspace(b,t,nticks)).astype(int))
        ax.set_yticklabels(np.round(np.linspace(b,t,nticks)).astype(int))
    else:
        ax.set_xticklabels(np.linspace(b,t,nticks))
        ax.set_yticklabels(np.linspace(b,t,nticks))
    pass
def minmax_norm(arr):
    norm_arr = (arr-arr.min())/(arr.max()-arr.min())
    return norm_arr
    
def plot_delta(d,h,b,t,Nm,ax=None,cmap='gray',**kwargs):
    '''
    Inputs: d: density contrast cube with Nm on a side
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    ax - plt axis to plot onto
    alpha - opacity
    cmap - colormap, default gray
    vmax - highest value it will plot. default 2.5 bc it looks nice.
    --------------------------------------------------
    Outputs: adds log(delta+1) plot to ax, returns slice of density.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    delta = d[idx,b_cou:t_cou,b_cou:t_cou]
    print('Shape of original delta array was: {}. Cropping array to a slice of shape: {}'.format(d.shape,delta.shape))
    im = ax.imshow(np.log10(delta.T+1.00001),origin='lower',interpolation=None,
                  cmap=cmap,extent=(b_cou,t_cou,b_cou,t_cou),**kwargs)
    cb = plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    set_window(b,t,Nm,ax)
    return delta

def plot_phi(p,h,b,t,Nm,ax=None,cmap='magma',logged=True,**kwargs):
    '''
    Inputs: p: grav potential cube with Nm on a side
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    ax - plt axis to plot onto
    alpha - opacity
    cmap - colormap, default gray
    logged - whether or not to normalize and plot log(phi) or just phi itself.
    --------------------------------------------------
    Outputs: adds log(phi+1) or phi plot to ax, returns slice of potential.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    phi = p[idx,b_cou:t_cou,b_cou:t_cou]
    print('Shape of original phi array was: {}. Cropping array to a slice of shape: {}'.format(p.shape,phi.shape))
    if logged == True:
        phi = minmax_norm(phi)
        im = ax.imshow(np.log10(phi.T+1.00001),origin='lower',interpolation=None,
                      cmap=cmap,
                      extent=(b_cou,t_cou,b_cou,t_cou),**kwargs)
    else:
        im = ax.imshow(phi.T,origin='lower',interpolation=None,
                      cmap=cmap,
                      extent=(b_cou,t_cou,b_cou,t_cou),**kwargs)
    cb = plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    set_window(b,t,Nm,ax)
    return phi

def plot_phi_conts(p,h,b,t,Nm,ax=None,cmap='viridis',fmt='str',normed=True,**kwargs):
    '''
    Inputs: p: grav potential cube with Nm on a side
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    ax - plt axis to plot onto
    alpha - opacity
    cmap - colormap, default gray
    fmt: formatter for contour labels. Either 'log' or 'str', log being 10^some power, and str
    being just numbers.
    normed: True for min-max normalizing phi, False for leaving as is
    --------------------------------------------------
    Outputs: adds contours of log(phi+1) or phi plot to ax, returns slice of potential.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    size_in_cou = t_cou - b_cou
    phi = p[idx,b_cou:t_cou,b_cou:t_cou]
    print('Shape of original phi array was: {}. Cropping array to a slice of shape: {}'.format(p.shape,phi.shape))
    if normed == True:
        phi = minmax_norm(phi)
    x = np.linspace(b_cou,t_cou,num=size_in_cou)
    [Y,Z] = np.meshgrid(x,x)
    if fmt == 'log':
        fmt = ticker.LogFormatterMathtext()
    else:
        fmt = ticker.StrMethodFormatter('{x:.2f}')
    fmt.create_dummy_axis()
    conts = ax.contour(Y,Z,phi.T,cmap=cmap,linewidths=1.6,**kwargs)
    ax.clabel(conts, conts.levels, inline=True, fmt=fmt, fontsize=10)
    set_window(b,t,Nm,ax)
    return conts

def plot_quiver(velx,vely,velz,h,b,t,Nm,ax=None,s=None,color='darkturquoise',stretch=None,**kwargs):
    '''
    Inputs: velx,y,z - velocity fields in each dimension with shape Nm^3
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    ax - plt axis to plot onto
    s - step to skip when plotting arrows. For large window sizes a step is recommended.
    color - color, default darkturquoise
    stretch - multiplies by a factor to overlay onto other plots with differing Nms.
    --------------------------------------------------
    Outputs: adds velocity vectors to ax, returns quiver object.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    size_in_cou = t_cou - b_cou
    x = np.linspace(b_cou,t_cou,num=size_in_cou)
    if stretch is not None:
        x = stretch * x
    [Y,Z] = np.meshgrid(x,x)
    vy = vely[idx,b_cou:t_cou,b_cou:t_cou]
    vz = velz[idx,b_cou:t_cou,b_cou:t_cou]
    if s is not None:
        arrow = ax.quiver(Y[::s,::s],Z[::s,::s],vy[::s,::s],vz[::s,::s],pivot='mid',
                          color=color,norm=Normalize(),**kwargs)
    else:
        arrow = ax.quiver(Y,Z,vy,vz,pivot='mid',color=color,norm=Normalize(),**kwargs)
    #set_window(b,t,Nm,ax)
    return arrow

def plot_stream(velx,vely,velz,h,b,t,Nm,ax=None,cmap=None,color='gold',density=1.0,stretch=None,**kwargs):
    '''
    Inputs: velx,y,z - velocity fields in each dimension with shape Nm^3
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    ax - plt axis to plot onto
    cmap - colormap. If provided, magnitude of velocity becomes colormapped.
    color - color, default gold
    density - density of streamlines, float.
    stretch - multiplies by a factor to overlay onto other plots with differing Nms.
    --------------------------------------------------
    Outputs: adds velocity streamlines to ax, returns stream object.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    size_in_cou = t_cou - b_cou
    x = np.linspace(b_cou,t_cou,num=size_in_cou)
    if stretch is not None:
        x = stretch * x
    [Y,Z] = np.meshgrid(x,x)
    vy = vely[idx,b_cou:t_cou,b_cou:t_cou]
    vz = velz[idx,b_cou:t_cou,b_cou:t_cou]
    if cmap is not None:
        color = np.sqrt(vy**2+vz**2)
    stream = ax.streamplot(Y,Z,vy,vz,color=color,cmap=cmap,density=density,**kwargs)
    #set_window(b,t,Nm,ax)
    return stream

# ORIGAMI things:
path_to_ori = '/home/idies/workspace/indra/origami/2_0_0/'
def readm(filename):
    '''
    Reads an origamitag .dat file. 
    0-void, 1-wall, 2-filament, 3-halo.
    '''
    F = open(filename,'r')
    num_part = np.fromfile(F,dtype=np.int32,count=1)[0]
    print('Total number of particles is: {}'.format(num_part))
    m = np.fromfile(F,dtype=np.int8,count=num_part)
    F.close()
    return m

# the old assign_ORI and plot_ORI functions are for when the database breaks
# the old fxns are slower though
def old_assign_ORI(pos,vel,ids,idx,Nm):
    '''
    Assigns particles to a the idx-th slice as well as vels, ids, and ORIGAMI tags.
    RUN128...can change file name in tags call.
    Inputs:
    pos: array of particle positions
    vel: array of particle vels
    ids: list of particle IDs
    idx: index to take the slice
    Nm: grid spacing 
    Returns: masked array of positions, velocities, ids, and tags,
    '''
    tags = readm(path_to_ori+'2_0_0_snap63_tag.dat')
    cell_len = 1000./Nm
    bot = cell_len*idx #'height' of slice in Mpc
    top = cell_len*(idx+1)
    indices = np.asarray(np.where((pos[:,0]>bot) & (pos[:,0]<top)))
    print('There are {} particles in slice {}'.format(len(pos[indices][0]),idx))
    return pos[indices][0], vel[indices][0], ids[indices][0], tags[indices][0]
def old_plot_ORI(pos,vel,ids,h,b,t,Nm,ax=None,alpha=0.7,s=5):
    '''
    Inputs: 
    pos - array of particle positions
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension (doesnt affect particles themselves but 
    needs to match Nm of delta/phi plotted underneath.)
    ax - plt axis to plot onto
    alpha - opacity
    s - subsample step to skip plotting all particles for time purposes.
    --------------------------------------------------
    Outputs: adds particles in a slice to ax, color-coded by ORI tag.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    pos_idx, vel_idx, ids_idx, tags_idx = old_assign_ORI(pos,vel,ids,idx,Nm)
    mask = np.where((pos_idx[:,1]>b)&(pos_idx[:,1]<t)&(pos_idx[:,2]>b)&(pos_idx[:,2]<t))
    pos_idx = pos_idx[mask]; tags_idx = tags_idx[mask]
    vel_idx = vel_idx[mask]; ids_idx = ids_idx[mask]
    pos_idx = pos_idx*(Nm/1000.) # to convert back to code units
    if s != 0:
        pos_idx = pos_idx[0::s]; tags_idx = tags_idx[0::s]
    oris = ax.scatter(pos_idx[:,1],pos_idx[:,2],c=tags_idx,s=0.5,
                   cmap=ListedColormap(["darkturquoise", "limegreen","yellow","red"]),
                   alpha=0.7)
    set_window(b,t,Nm,ax)
    return oris

def assign_ORI(h,b,t,Nm):
    '''
    Could add getvels=True to particlesInShape() call if wanted.
    '''
    idx = Mpc_to_cou(h,Nm)
    tags = readm(path_to_ori+'2_0_0_snap63_tag.dat')
    cell_len = 1000./Nm
    shape = indra.Box(h,b,b,h+cell_len,t,t)
    runid = 128; snapnum = 63 # change if want to look at other Indra runs
    box_ps = indra.particlesInShape(runid,snapnum,shape,getIDs=True)
    tags = tags[box_ps['ids']]
    print('There are {} particles in slice {}'.format(box_ps['NumParticles'],idx))
    box_pos = np.column_stack([box_ps['x'],box_ps['y'],box_ps['z']])
    return box_pos, box_ps['ids'], tags

def plot_ORI(pos,vel,ids,h,b,t,Nm,ax=None,s=5,**kwargs):
    '''
    Inputs: 
    pos - array of particle positions
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension (doesnt affect particles themselves but 
    needs to match Nm of delta/phi plotted underneath.)
    ax - plt axis to plot onto
    s - subsample step to skip plotting all particles for time purposes.
    --------------------------------------------------
    Outputs: adds particles in a slice to ax, color-coded by ORI tag.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    pos_idx, ids_idx, tags_idx = assign_ORI(h,b,t,Nm)
    pos_idx = pos_idx*(Nm/1000.) # to convert back to code units
    if s != 0:
        pos_idx = pos_idx[0::s]; tags_idx = tags_idx[0::s]
    oris = ax.scatter(pos_idx[:,1],pos_idx[:,2],c=tags_idx,s=0.5,
                   cmap=ListedColormap(["darkturquoise", "limegreen","yellow","red"]),
                   **kwargs)
    set_window(b,t,Nm,ax)
    return oris

def plot_ORI_cts(a,h,b,t,Nm,ax=None,alpha=1,stretch=None,level=None,cmap='viridis',**kwargs):
    '''
    Inputs: a: array cube with Nm on a side of % ORIGAMI tag abundance
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    label - which ORIGAMI abundance we are displaying (void,wall,fila,halo)
    ax - plt axis to plot onto
    alpha - opacity
    stretch - factor to stretch by if plotting something with differing Nm.
    level - if not None, plot only voxels that are above some certain level
    of percentage void (w,f,h) particles
    --------------------------------------------------
    Outputs: adds image of abundance in a slice to ax, returns slice of array.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    arr = a[idx,b_cou:t_cou,b_cou:t_cou]
    print('Shape of original array was: {}. Cropping array to a slice of shape: {}'.format(a.shape,arr.shape))
    if level is not None:
        arr = np.ma.masked_array(arr, arr < level)
    if stretch is not None:
        im = ax.imshow(arr.T,origin='lower',
                       extent=(stretch*b_cou,stretch*t_cou,stretch*b_cou,stretch*t_cou),
                       interpolation=None,alpha=alpha,cmap=cmap,**kwargs)
    else:
        im = ax.imshow(arr.T,origin='lower',extent=(b_cou,t_cou,b_cou,t_cou),
                       interpolation=None,alpha=alpha,cmap=cmap,**kwargs)
    cb = plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    set_window(b,t,Nm,ax)
    return arr
    
def plot_eig_signs(eigs,h,b,t,Nm,ax=None,alpha=0.8,stretch=None):
    '''
    Inputs: 
    eigs - array of signs of eigenvalues of the Hessian of the potential. 0 means mixed signs,
    1 means (+ + +), 2 means (- - -) (local maxima). 
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension (doesnt affect particles themselves but 
    needs to match Nm of delta/phi plotted underneath.)
    ax - plt axis to plot onto
    alpha - opacity
    --------------------------------------------------
    Outputs: adds image of eig signs in a slice to ax, color-coded as clear=mixed,
    red (+ + +), cyan (- - -).
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    eigs = eigs[idx,b_cou:t_cou,b_cou:t_cou]
    eigs = np.ma.masked_array(eigs, eigs < 1.0)
    if stretch is not None:
        signs = ax.imshow(eigs.T,cmap=ListedColormap(['r','c']),
                          origin='lower',extent=(stretch*b_cou,stretch*t_cou,stretch*b_cou,stretch*t_cou),
                          interpolation=None,alpha=alpha)
    else:
        signs = ax.imshow(eigs.T,cmap=ListedColormap(['r','c']),
                          origin='lower',extent=(b_cou,t_cou,b_cou,t_cou),
                          interpolation=None,alpha=alpha)
    cb = plt.colorbar(signs,ax=ax,fraction=0.046, pad=0.04)
    return signs

def plot_3D_phi(p,eigs,h,b,t,Nm,ax=None,plot_eigs=True):
    '''
    Inputs: 
    phi - grav potential array w/ Nm on a side
    eigs - eig sign array
    h - height of slice
    b,t - lower/upper bound of window
    Nm - # of mesh cells in one dimension
    ax - plt axis to plot onto
    plot_eigs - if True, will plot eig signs underneath phi surface
    --------------------------------------------------
    Outputs: plots a slice of phi as a surface.
    '''
    if ax is None:
        ax = plt.gca()
    ax = plt.axes(projection='3d')
    runid = 128
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    size_in_cou = t_cou - b_cou
    phi = p[idx,b_cou:t_cou,b_cou:t_cou]
    x = np.linspace(b_cou,t_cou,num=size_in_cou)
    [Y,Z] = np.meshgrid(x,x)
    surf = ax.plot_surface(Y,Z,phi.T,cmap='inferno')
    ax.set_title(r'Moving through Indra run {} with $\Phi$ as Z axis [slice {}]'.format(runid,idx))
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel(r'$\Phi$')
    if plot_eigs == True:
        idx = Mpc_to_cou(h,512) # H_eig_signs has a Nm = 512
        b_cou = Mpc_to_cou(b,512); t_cou = Mpc_to_cou(t,512)
        size_in_cou = t_cou - b_cou
        eigs = eigs[idx,b_cou:t_cou,b_cou:t_cou]
        b_cou *= 2; t_cou *= 2 # stretch to accomodate phi1024
        x = np.linspace(b_cou,t_cou,num=size_in_cou)
        [Y,Z] = np.meshgrid(x,x)
        floor = ax.get_zlim()[0]
        cset = ax.contourf(Y,Z,eigs.T,zdir='z',offset=floor,
                           cmap=ListedColormap(['snow','r','c']),
                          alpha=0.66)
    return surf

def plot_tidal_mask(m,h,b,t,Nm,ax=None,**kwargs):
    '''
    Plots a slice of the tidal mask as defined in Hahn/Porciani 2017. 
    Void cells are 0s, walls 1s, filaments 2s, halos 3s.
    Void - black, wall - green, fila - yellow, halo - red
    Inputs: m - mask array with shape (Nm,Nm,Nm)
    h,b,t - height and window limits in h-1 Mpc
    Nm - # of grid cells on a side
    
    Outputs:
    returns imshow of tidal mask, plots an imshow
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    m = m[idx,b_cou:t_cou,b_cou:t_cou]
    im = ax.imshow(m.T,origin='lower',interpolation=None,
                  extent=(b_cou,t_cou,b_cou,t_cou),
                  cmap=ListedColormap(['k','g','y','r']),
                  **kwargs)
    cb = plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    set_window(b,t,Nm,ax)
    return im

def plot_gen_arr(a,h,b,t,Nm,ax=None,**kwargs):
    '''
    FOR INDRA
    Plot any array with window, cropping standard to other
    plotter fxns.
    '''
    if ax is None:
        ax = plt.gca()
    idx = Mpc_to_cou(h,Nm)
    b_cou = Mpc_to_cou(b,Nm); t_cou = Mpc_to_cou(t,Nm)
    a = a[idx,b_cou:t_cou,b_cou:t_cou]
    im = ax.imshow(a.T,origin='lower',interpolation=None,
                  extent=(b_cou,t_cou,b_cou,t_cou),
                  **kwargs)
    cb = plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    set_window(b,t,Nm,ax)
    return im

def plot_arr(a,idx,ax=None,logged=False,pct=100,cb=True,segmented_cb=False,cmap='magma',**kwargs):
    '''
    For plotting any array. 
    a: 3d array
    idx: index of slice to plot
    ax: axis to plot onto
    logged: whether or not to plot log10 of array
    pct: percent of array to plot (from bottom left-hand corner). default 100%.
    cb: whether or not to plot colorbar
    segmented_cb: set to true when plotting multi-label tidal mask to get 
    discrete color values instead of a continuous gradient.
    '''
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
    a = a[idx]
    ext = round(a.shape[0]*(pct/100.))
    a = a[0:ext,0:ext]
    if segmented_cb:
        # default to magma? can change this later
        cmap = plt.get_cmap(cmap,4)
        if logged == True:
            im = ax.imshow(np.log10(a),origin='lower',interpolation=None,cmap=cmap,
            **kwargs)
        else:
            im = ax.imshow(a,origin='lower',interpolation=None,cmap=cmap,
            **kwargs)
    else:
        if logged == True:
            im = ax.imshow(np.log10(a),origin='lower',interpolation=None,cmap=cmap,
            **kwargs)
        else:
            im = ax.imshow(a,origin='lower',interpolation=None,cmap=cmap,
            **kwargs)
    if cb:
        if segmented_cb:
            tick_pos = np.linspace(0.75,3,4)-0.75/2
            cb = plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04,ticks=tick_pos)
            cb.ax.set_yticklabels(['Void','Wall','Filament','Halo'])
        else:
            cb = plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    return im

# Create fxn for showing 0s in a binary image as transparent using masked array:
def alpha0(arr):
    arr = np.ma.masked_where(arr == 0, arr)
    return arr

#---------------------------------------------------------
# Summary statistics for a Numpy array
#---------------------------------------------------------
def summary(array):
  print('### Summary Statistics ###')
  print('Shape: ',str(array.shape))
  print('Mean: ',np.mean(array))
  print('Median: ',np.median(array))
  print('Maximum: ',np.max(array))
  print('Minimum: ',np.min(array))
  print('Std deviation: ',np.std(array))
# ANIMATION ###
from PIL import Image
def generate_gif(arr,filename,fps=10):
    '''
    Input: arr NumPy array of floats.
    filename str filename to save gif as.
    cmap_name: str gray, viridis, magma, etc.
    fps: int frames per second
    Output: saves a gif scan.
    '''
    imgs = [arr[i] for i in range(arr.shape[0])]
    imgs = [Image.fromarray(img) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=fps**-1., loop=0)
    print(f'Saved gif at {filename}')

import imageio
def save_slices_gif(arr, cmap_name, filename, overlay_arr=None, alpha=0.5,fps=50):
    fig, ax = plt.subplots()

    cmap = plt.get_cmap(cmap_name)

    images = []
    for i in range(arr.shape[0]):
        ax.clear()

        # Plot the main image
        ax.imshow(arr[i], cmap=cmap)
        ax.axis('off')

        if overlay_arr is not None:
            # Create a mask for the overlay array where the values are zero
            mask = np.ma.masked_where(overlay_arr[i] == 0, overlay_arr[i])
            # Create an RGBA array with the color map and the alpha channel
            rgba = cmap(mask)
            rgba[:, :, 3] = alpha
            # Plot the RGBA array on top of the main image
            ax.imshow(rgba)

        fig.canvas.draw()

        # Convert the canvas to a NumPy array
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
        images.append(image)

    # Save the images as a GIF
    imageio.mimsave(filename, images, fps=fps)

def show_image_row(imgs, plot_size, **kwargs):
    '''
    Function that takes in a list of images and plots them in a row.
    plot size is a list of two numbers, the first being the width of the plot
    and the second being the height.
    **kwargs examples include cmap, vmin, vmax, interpolation, etc. 
    '''
    fig, ax = plt.subplots(1,len(imgs), figsize=(plot_size[0],plot_size[1]),tight_layout=True)
    for i in range(len(imgs)):
        ax[i].axis('off')
        _ = ax[i].imshow(imgs[i], origin='lower', **kwargs)

#---------------------------------------------------------
# history.history plotting
# plot_training_metric_to_ax: plots a training metric to an axis
# plot_training_metrics_all: plots all training metrics
#---------------------------------------------------------
def plot_training_metric_to_ax(ax,history,metric,GRID=True,CSV_FLAG=False,**kwargs):
    '''
    Function to plot a metric from history.history. Special cases for
    'loss' and 'acc' are included. Now handles metrics with output prefixes
    (e.g., 'last_activation_loss', 'lambda_output_loss').
    ax is the axis to plot on.
    history is the history object from model.fit.
    GRID is a boolean to determine if grid is plotted.
    CSV_FLAG: bool, for plotting training metrics from a CSV file.
    '''
    if not CSV_FLAG:
        metric_dict = history.history
        available_keys = list(metric_dict.keys())
    else:
        metric_dict = history
        available_keys = list(metric_dict.keys())
    
    # Helper function to find metric keys that match a pattern
    def find_metric_keys(pattern, exclude_val=False):
        keys = []
        for key in available_keys:
            if exclude_val and key.startswith('val_'):
                continue
            if pattern in key and not key.startswith('val_'):
                keys.append(key)
        return keys
    
    def find_val_metric_keys(pattern):
        keys = []
        for key in available_keys:
            if key.startswith('val_') and pattern in key:
                keys.append(key)
        return keys
    
    # Special handling for loss and accuracy
    if metric == 'loss' or metric == 'val_loss':
        # Find all loss-related keys
        train_loss_keys = find_metric_keys('loss', exclude_val=True)
        val_loss_keys = find_val_metric_keys('loss')
        
        if train_loss_keys and val_loss_keys:
            # Use the first main loss (usually the total loss)
            train_key = train_loss_keys[0] if 'loss' in train_loss_keys else train_loss_keys[0]
            val_key = val_loss_keys[0] if 'val_loss' in val_loss_keys else val_loss_keys[0]
            
            epochs = len(metric_dict[train_key])
            full_epochs = np.arange(0, epochs)
            
            ax.plot(full_epochs, np.array(metric_dict[train_key]).astype(float), label='Train', **kwargs)
            ax.plot(full_epochs, np.array(metric_dict[val_key]).astype(float), label='Test', **kwargs)
            ax.set_title('Model Loss')
            ax.legend()
        else:
            print(f"Warning: Could not find loss metrics. Available keys: {available_keys}")
            return
            
    elif metric == 'accuracy' or metric == 'val_accuracy':
        # Find all accuracy-related keys
        train_acc_keys = find_metric_keys('accuracy', exclude_val=True)
        val_acc_keys = find_val_metric_keys('accuracy')
        
        if train_acc_keys and val_acc_keys:
            # Prefer main accuracy, then any accuracy metric
            train_key = 'accuracy' if 'accuracy' in train_acc_keys else train_acc_keys[0]
            val_key = 'val_accuracy' if 'val_accuracy' in val_acc_keys else val_acc_keys[0]
            
            epochs = len(metric_dict[train_key])
            full_epochs = np.arange(0, epochs)
            
            ax.plot(full_epochs, np.array(metric_dict[train_key]).astype(float), label='Train', **kwargs)
            ax.plot(full_epochs, np.array(metric_dict[val_key]).astype(float), label='Test', **kwargs)
            ax.set_title('Model Accuracy')
            ax.legend()
        else:
            print(f"Warning: Could not find accuracy metrics. Available keys: {available_keys}")
            return
    else:
        # For other metrics, find both train and validation versions
        train_keys = find_metric_keys(metric, exclude_val=True)
        val_keys = find_val_metric_keys(metric)
        
        if train_keys:
            # Use the best matching training key
            if metric in train_keys:
                train_key = metric
            else:
                train_key = train_keys[0]
                print(f"Using '{train_key}' for requested metric '{metric}'")
            
            epochs = len(metric_dict[train_key])
            full_epochs = np.arange(0, epochs)
            train_mask = np.isfinite(np.array(metric_dict[train_key]).astype(np.double))
            
            ax.plot(full_epochs[train_mask], np.array(metric_dict[train_key]).astype(np.double)[train_mask], 
                   label='Train', **kwargs)
            
            # Plot validation if available
            if val_keys:
                # Find corresponding validation key
                val_key = None
                for vkey in val_keys:
                    if metric in vkey or train_key.replace('val_', '') in vkey:
                        val_key = vkey
                        break
                if not val_key:
                    val_key = val_keys[0]
                
                val_mask = np.isfinite(np.array(metric_dict[val_key]).astype(np.double))
                ax.plot(full_epochs[val_mask], np.array(metric_dict[val_key]).astype(np.double)[val_mask], 
                       label='Test', **kwargs)
            
            ax.set_title(f'Model {metric}')
            if val_keys:
                ax.legend()
        else:
            print(f"Warning: Could not find metric '{metric}'. Available keys: {available_keys}")
            return
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    if GRID:
        ax.grid()

# create function to plot every single available metric from history.history:
def plot_training_metrics_all(history,FILE_OUT,aspect='rect',savefig=False,CSV_FLAG=False,**kwargs):
    '''
    Function to plot all available metrics from history.history. 
    Now handles metrics with output prefixes (e.g., 'last_activation_loss').
    FILE_OUT is the filepath to save the plot. 
    aspect is either 'rect' or 'square'.
    If savefig=True, the plot will be saved.
    CSV_FLAG: bool, for plotting training metrics from a CSV file.
    '''
    if not CSV_FLAG:
        all_keys = list(history.history.keys())
    else:
        all_keys = list(history.keys())
    
    print(f"Available metric keys: {all_keys}")
    
    # Skip validation metrics and specific keys
    skip_patterns = ['val_', 'epoch']
    if CSV_FLAG:
        skip_patterns.append('epoch')
    
    # Filter out validation metrics and other unwanted keys
    filtered_keys = []
    for key in all_keys:
        should_skip = False
        for pattern in skip_patterns:
            if key.startswith(pattern):
                should_skip = True
                break
        if not should_skip:
            filtered_keys.append(key)
    
    # Group similar metrics and pick representative ones
    # Priority: main metrics first, then output-specific ones
    metric_groups = {
        'loss': [],
        'accuracy': [],
        'f1_micro': [],
        'mcc': [],
        'void_f1': [],
        'other': []
    }
    
    # Categorize metrics
    for key in filtered_keys:
        categorized = False
        for metric_type in ['loss', 'accuracy', 'f1_micro', 'mcc', 'void_f1']:
            if metric_type in key:
                metric_groups[metric_type].append(key)
                categorized = True
                break
        if not categorized:
            metric_groups['other'].append(key)
    
    # Select the best representative for each group
    metrics_to_plot = []
    for group_name, group_keys in metric_groups.items():
        if group_keys:
            # For main metrics, prefer the simple name, otherwise take the first
            if group_name in ['loss', 'accuracy']:
                if group_name in group_keys:
                    metrics_to_plot.append(group_name)
                else:
                    metrics_to_plot.append(group_keys[0])
            else:
                # For other metrics, prefer the simplest name (shortest)
                best_key = min(group_keys, key=len)
                metrics_to_plot.append(best_key)
    
    # Add any remaining metrics from 'other' category
    metrics_to_plot.extend(metric_groups['other'])
    
    # Remove duplicates while preserving order
    metrics_to_plot = list(dict.fromkeys(metrics_to_plot))
    
    print(f"Plotting metrics: {metrics_to_plot}")
    
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        print("Warning: No metrics found to plot!")
        return
    
    if aspect == 'square':
        # if we want a square grid of plots or as close to it as possible:
        n_cols = int(np.ceil(np.sqrt(n_metrics))); n_rows = int(np.ceil(n_metrics/n_cols))
        figsize = (15,15)
    if aspect == 'rect':
        # if we want a grid of plots with 3 columns:
        n_cols = 3; n_rows = int(np.ceil(n_metrics/n_cols))
        figsize = (13,20)
    
    fig, ax = plt.subplots(n_rows,n_cols,figsize=figsize,layout='constrained')
    
    # Handle case where we only have one subplot
    if n_metrics == 1:
        ax = [ax]
    elif n_rows == 1 or n_cols == 1:
        # ax is already a 1D array
        pass
    else:
        # ax is a 2D array, flatten it
        ax = ax.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        try:
            if n_rows == 1 and n_cols == 1:
                plot_training_metric_to_ax(ax[0], history, metric, CSV_FLAG=CSV_FLAG, **kwargs)
            else:
                plot_training_metric_to_ax(ax[i], history, metric, CSV_FLAG=CSV_FLAG, **kwargs)
        except Exception as e:
            print(f"Error plotting metric '{metric}': {e}")
            continue
    
    # remove any empty axes:
    if n_metrics < len(ax):
        for i in range(n_metrics, len(ax)):
            fig.delaxes(ax[i])
    
    if savefig:
        plt.savefig(FILE_OUT)
        print(f"Training metrics plot saved to {FILE_OUT}")
    else:
        plt.show()

# returns golden ratio sized rectangle for plotting:
def golden_ratio(width):
    '''
    Returns the height of a golden ratio sized rectangle for plotting.
    '''
    GR = (1 + np.sqrt(5))/2
    return width/GR