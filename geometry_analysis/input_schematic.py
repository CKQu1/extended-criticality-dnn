import numpy as np
from tqdm import tqdm

import sys
import os
import re

from mpl_toolkits.mplot3d import axes3d

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old

# ----- plot -----

def input_plot():


    import matplotlib as mpl
    from time import time
    t0 = time()

    from matplotlib import pyplot as plt
    from matplotlib import cm
    # colorbar
    cm_type = 'twilight'
    plt.rcParams["font.family"] = "serif"     # set plot font globally

    plot_path = "/project/PDLAI/project2_data/figure_ms"
 
    # Font size
    tick_size = 16.5
    label_size = 16.5
    axis_size = 16.5
    legend_size = 14
    linewidth = 0.8

    # great circle input
    N = 2000
    N_thetas = 2000
    q_fixed = 1

    # Generate circular manifold.
    hs = np.zeros([N, N_thetas])
    thetas = np.linspace(0, 2*np.pi, N_thetas)
    x = q_fixed * np.cos(thetas)
    y = q_fixed * np.sin(thetas)
    z = np.array([0]*len(y))

    # rotate
    """
    rangle = 90
    rotation_mat = np.array([[0,np.cos(rangle),0],
                             [np.cos(rangle),0,-np.sin(rangle)],
                             [np.sin(rangle),0,np.cos(rangle)]]
                             )
    circle_plot = rotation_mat @ np.stack([x,y,z])
    x = circle_plot[0,:]
    y = circle_plot[1,:]
    z = circle_plot[2,:]
    """

    #fig = plt.figure(figsize=(9.5,7.142))
    #fig = plt.figure(figsize=(9.5,7.142/3 + 0.75))
    fig = plt.figure(figsize=(15,15))
    gs = mpl.gridspec.GridSpec(10, 10, wspace=0, hspace=0)
    cmap_bd = [0, 2*np.pi]

    ax = fig.add_subplot(gs[:, :],projection='3d')
    im = ax.scatter(z , y , x, c=thetas, vmin=cmap_bd[0], vmax=cmap_bd[1], marker='.', s=200, alpha=1, cmap=cm.get_cmap(cm_type))
    # lines
    #ax.plot(x , y , z,color='k', zorder=0, linewidth=0.25, alpha=.35)
    #ax.zaxis.set_rotate_label(False) 
    ax.view_init(0, 80)

    ax.set_aspect('auto')
    ax.set_axis_off()

    # set lims
    #pmax = 0.05
    #pmax = 0.06
    #ax.set_xlim(-pmax, pmax); ax.set_ylim(-pmax, pmax);
    #ax.set_zlim(-pmax, pmax);

    # tick labels
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_zticklabels([]);

    plt.savefig(f"{plot_path}/proj3d_input.pdf", bbox_inches='tight')
    print(f"Input schematic done!")
    print(f"{time() - t0} s!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
