import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
import random
import seaborn as sns
import scipy
import scipy.io as sio
import scipy.stats as sst
import sys
import time
import torch

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from ast import literal_eval
from os.path import join, isfile
from scipy.stats import levy_stable, norm, lognorm
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions
from tqdm import tqdm

print(scipy.__version__)
print(os.getcwd())

def replace_name(weight_name,other):
    assert isinstance(other,str)
    ls = weight_name.split("_")
    ls[-3] = other
    #ls += other
    return '_'.join(ls)

# fitting and testing goodness of fit
def fit_and_test(data, dist_type):

    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        params = pconv(*levy_stable._fitstart(data))
        r = levy_stable.rvs(*params, size=len(data))
        logl = np.sum(np.log(levy_stable.pdf(data, *params)))
    elif dist_type == 'normal':
        params = distributions.norm.fit(data)
        r = norm.rvs(*params, len(data))
        logl = np.sum(np.log(norm.pdf(data, *params)))
    elif dist_type == 'tstudent':
        params = sst.t.fit(data)
        r = sst.t.rvs(*params, len(data))
        logl = np.sum(np.log(sst.t.pdf(data, *params)))
    elif dist_type == 'lognorm':
        params = lognorm.fit(data)
        r = lognorm.rvs(*params, size=len(data))
        logl = np.sum(np.log(lognorm.pdf(data, *params)))

    # statistical tests
    # AD test
    try:
        ad_test = anderson_ksamp([r, data])
        ad_siglevel = ad_test.significance_level
    except:
        ad_siglevel = None
        pass

    # KS test
    try:
        ks_test = ks_2samp(r, data, alternative='two-sided')
        ks_stat = ks_test.statistic
        ks_pvalue = ks_test.pvalue
    except:
        ks_stat, ks_pvalue = None, None
        pass

    if dist_type == 'normal':
        shapiro_test = shapiro(data)
        shapiro_stat = shapiro_test[0]
        shapiro_pvalue = shapiro_test[1]
        return params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue
    else:
        return params, logl, ad_siglevel, ks_stat, ks_pvalue

# Stable fit function
pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

t0 = time.time()

# Loading weight matrix ----------------------
weight_path = os.getcwd()
weight_name = "alexnet_layer_1_2"
wmat_idx = 2

col_names = ['wmat_idx','w_size', 'fit_size',
              'alpha','beta','delta','sigma', 'logl_stable', 'ad sig level stable', 'ks stat stable', 'ks pvalue stable',                        # stable(stability, skewness, location, scale), 3 - 10
              'mu', 'sigma_norm', 'logl_norm', 'ad sig level normal','ks stat normal', 'ks pvalue normal', 'shap stat', 'shap pvalue',           # normal(mean, std), 11 - 18
              'nu', 'sigma_t', 'mu_t', 'logl_t', 'ad sig level tstudent','ks stat tstudent', 'ks pvalue tstudent',                               # tstudent(dof, scale, location), 19 - 25
              'shape_lognorm', 'loc_lognorm', 'scale_lognorm', 'logl_lognorm', 'ad sig level lognorm','ks stat lognorm', 'ks pvalue lognorm'     # lognormal(loc, scale), 26 - 32
              ]

df = pd.DataFrame(np.zeros((1,len(col_names))))
df = df.astype('object')
df.columns = col_names

weights = torch.load(f"{weight_path}/{weight_name}")
weights = weights.detach().numpy()
w_size = len(weights)

# 1. values much smaller than zero are filtered out
weights = weights[np.abs(weights) >0.00001]

print(f"True size: {w_size}")
print(f"Fit size: {len(weights)}")

# save params
df.iloc[0,0:3] = [wmat_idx, w_size, len(weights)]

# Fitting
index_dict = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
for dist_type in tqdm(["levy_stable", "normal", "tstudent", "lognorm"]):
    idxs = index_dict[dist_type]
    if dist_type == "normal":
        params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue = fit_and_test(weights, dist_type)
        df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue]
    else:
        params, logl, ad_siglevel, ks_stat, ks_pvalue = fit_and_test(weights, dist_type)
        df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue]

    df.iloc[0,idxs[0]:idxs[1]] = list(params)
    #print(f"{dist_type} done!")
    #print('\n')

# Save params ----------------------
data_name = replace_name(weight_name,'allfit')
df.to_csv(f'{weight_path}/{data_name}.csv', index=False)
print("df saved!")
print(df)

# ----- plotting -----

print(f"Min weight: {weights.min()}, Max weight: {weights.max()}")
percentiles = [np.percentile(weights, per) for per in [5e-6, 50, 50, 99.999995]]
pl1, pl2, pu1, pu2 = percentiles
print(percentiles)
# Plots ----------------------
fig, axs = plt.subplots(1, 3, sharex = False,sharey=False,figsize=(12.5 + 4.5, 9.5/3 + 2.5))
# plot 1 (full distribution)
axs[0].hist(weights, bins=1000, density=True)
wmin, wmax = weights.min(), weights.max()
bd = min(np.abs(pl1), np.abs(pu2))
x = np.linspace(-bd, bd, 1000)
axs[0].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
axs[0].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
axs[0].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
axs[0].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)
axs[0].legend(loc = 'upper right')
axs[0].set_xlim(-bd, bd)

# plot 2 (log-log hist right tail)
axs[1].hist(weights, bins=2000, histtype="step")
sns.kdeplot(weights, shade=True, color='blue', ax=axs[1])

#x = np.linspace((mean_gaussian + wmax)/2, wmax, 500)
x = np.linspace(pu1, pu2, 500)
axs[1].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
axs[1].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
axs[1].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
axs[1].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)
axs[1].set_xlim(pu1,pu2)
axs[1].set_xscale('symlog'); axs[1].set_yscale('log')

# plot 3 (left tail)
axs[2].hist(weights, bins=2000, histtype="step")
sns.kdeplot(weights, shade=True, color='blue', ax=axs[2])

x = np.linspace(pl1, pl2, 500)
axs[2].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
axs[2].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
axs[2].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
axs[2].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)
axs[2].set_xlim(pl1,pl2)
axs[2].set_xscale('symlog'); axs[2].set_yscale('log')

print("Starting plot")
plot_title = str(list(df.iloc[0,0:3])) + '\n'
plot_title += "Levy" + str(["{:.2e}".format(num) for num in df.iloc[0,3:7]]) + '  '
plot_title += "Normal" + str(["{:.2e}".format(num) for num in df.iloc[0,11:13]]) + '\n'
plot_title += "TStudent" + str(["{:.2e}".format(num) for num in df.iloc[0,19:22]]) + '  '
plot_title += "Lognorm" + str(["{:.2e}".format(num) for num in df.iloc[0,26:29]])
plt.suptitle(plot_title)
plot_name = replace_name(weight_name,'plot')
#plt.savefig(f"{weight_path}/{plot_name}.pdf", bbox_inches='tight', format='pdf')
#plt.clf()
t_last = time.time()
print(f"{weight_name} done in {t_last - t0} s!")
plt.show()
# Time