        import matplotlib.pyplot as plt

# reference: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

# change font
pub_font = {'family' : 'serif'}
#plt.rc('font', **pub_font)

# set up figures font sizes 
def plot_sizes(small:bool):
    if small == True:
        title_size = 26.5
        tick_size = 26.5
        label_size = 26.5
        axis_size = 26.5
        legend_size = 23.5
    else:
        title_size = 16.5    
        tick_size = 13
        label_size = 16.5
        axis_size = 16.5
        legend_size = 14
        linewidth = 0.8

    params = {'legend.fontsize': legend_size,
              #'figure.figsize': (20,8),
              'axes.labelsize': tick_size,
              'axes.titlesize': title_size,
              'xtick.labelsize': tick_size,
              'ytick.labelsize': tick_size}

    return params

# allows to fix axes height and width 
# (https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units)
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

#plt.rcParams.update(params)

"""
plt.rc('axes', titlesize=tick_size)     # fontsize of the axes title
plt.rc('axes', labelsize=tick_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=title_size)  # fontsize of the figure title
"""
    
