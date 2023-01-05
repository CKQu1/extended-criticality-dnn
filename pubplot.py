import matplotlib.pyplot as plt

# reference: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

# change font
pub_font = {'family' : 'serif'}
#plt.rc('font', **pub_font)

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

#plt.rcParams.update(params)

"""
plt.rc('axes', titlesize=tick_size)     # fontsize of the axes title
plt.rc('axes', labelsize=tick_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=title_size)  # fontsize of the figure title
"""
    
