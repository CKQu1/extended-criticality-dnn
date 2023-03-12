import argparse
import scipy.io as sio
import math
import matplotlib
# load latex
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import numpy as np
import time
import torch
import os
import pandas as pd
import string

from ast import literal_eval
from os.path import join
from scipy.stats import levy_stable, norm, distributions
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator

from pretrained_wfit import replace_name

pub_font = {'family' : 'serif'}
plt.rc('font', **pub_font)

t0 = time.time()

# ---------------------------

# 3 by 3 template

inset_width, inset_height = .1, .13

tick_size = 18.5
label_size = 18.5
axis_size = 18.5
legend_size = 14
linewidth = 0.8
text_size = 14
mark = 1/3
width = 0.26
height = 0.33
inset_x = 0.23
inset_y1, inset_y2 = .52, .85
label_ls = [f"({letter})" for letter in list(string.ascii_lowercase)]
linestyle_ls = ["solid", "dashed","dashdot"]

params = {'legend.fontsize': legend_size,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size}
plt.rcParams.update(params)

fig, axs = plt.subplots(3, 3, sharex = False,sharey=False,figsize=(12.5 + 4.5, 9.5/2*3))

for i in range(3):

    axis = axs[0,i]

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    if i == 2:
        # colour bar
        #cbar_ax = plt.axes([0.93, 0.5, 0.012, height])
        phase_scatter = axis.scatter([1,2,3], [1,2,3], c=[1,2,3], s=50, cmap="plasma")
        cbar_ax = plt.axes([0.7, 1.0, 0.28, 0.01])
        cbar_ax.xaxis.set_ticks_position('bottom')
        cbar = fig.colorbar(phase_scatter, ax=axis, 
                            cax=cbar_ax, orientation="horizontal", 
                            panchor=False)

    if i == 0:
        axis_inset = plt.axes([inset_x, inset_y1, inset_width, inset_height], xscale='log', yscale='log')

fcn_axs = [] 
for i in range(3):

    #axis = plt.axes([mark*(i%3), 0, width, height])
    axis = axs[1,i]

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)  

    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)

    fcn_axs.append(axis)  


# inset plot 2
axis_inset = plt.axes([inset_x, inset_y2, inset_width, inset_height], xscale='log', yscale='log')
lb, ub = 0.003, 0.0065

axis_inset.set_xlim(lb,ub)
axis_inset.set_ylim(8e-1,3e2)

axis_inset.minorticks_off()
axis_inset.set_xticklabels([])
axis_inset.set_yticklabels([])

axis_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

plt.subplots_adjust(hspace=0.2)
plt.tight_layout()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
#fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
plt.savefig(f"{fig1_path}/pretrained_template.pdf", bbox_inches='tight')

