import argparse
import math
import numpy as np
import os
import pandas as pd
import random
import scipy
import scipy.io as sio
import sys
import time

import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy import optimize
from scipy.optimize import root, fsolve

# fixed pseudo length of the FCN layers (no bias)
def q_star(w_alpha, w_mult):
    
    alpha = w_alpha 
    beta = 0
    loc = 0
    phi=np.tanh

    # for reference
    #scale_multiplier =  w_mult
    #scale = (1/(2*math.sqrt(layer_dim[0] * layer_dim[1])))**(1/alpha)
    #new_weights = levy_stable.rvs(alpha, beta, loc, scale * scale_multiplier, size=layer_dim)

    # find root of fixed point equation
    #q = levy_stable.expect(np.tanh(z), args = (alpha, beta), loc = loc, scale = q/2)

    def expectation_term(x):
        return np.abs(np.tanh(x))**w_alpha
    
    D_w = w_mult**(1/alpha)
    
    #integral_minus_q = lambda q: D_w * levy_stable.expect(expectation_term, args = (alpha, beta), loc = loc, scale = q/2) - q
    integral_minus_q = lambda q: D_w * levy_stable.expect(expectation_term, args = (alpha, beta), loc = loc, scale = (q/2)**(1/alpha)) - q
    q_fixed = fsolve(integral_minus_q, 5)

    #q_fixed = scipy.optimize.root_scalar(integral_minus_q, bracket = [1e-3, 100])

    return q_fixed

# test cases

"""
#integral = lambda q: Dw * scipy.stats.levy_stable(w_alpha,0,scale=(q*0.5)**(1./alpha)).expect(lambda x: abs(phi(x))**alpha)
#q_ls = np.linspace(0, 15, 50)
#integral_ls = [integral(q) for q in q_ls]
#plt.plot(q_ls, integral_ls, label = "Eval")
#plt.plot(q_ls, q_ls)
#plt.plot(q_fixed, q_fixed, 'ro')
#plt.show()

t0 = time.time()

w_alpha = 2 
beta = 0
loc = 0
w_mult = 1.2
D_w = w_mult**(1/w_alpha)

q_fixed = q_star(w_alpha, w_mult)
print(rf"$q^*$: {q_fixed}")

def abs_tanh_alpha(x):
    return abs(math.tanh(x))**w_alpha
    #return abs(math.tanh(x))

# plotting

integral = lambda q: D_w * levy_stable.expect(abs_tanh_alpha, args = (w_alpha, beta), loc = loc, scale = q/2)

q_ls = np.linspace(0, 15, 50)
integral_ls = []
for q in q_ls:

    e = integral(q)
    integral_ls.append(e)

plt.plot(q_ls, integral_ls, label = "Eval")
plt.plot(q_ls, q_ls)
plt.plot(q_fixed, q_fixed, 'ro')

plt.legend()

print(f"Time: {time.time() - t0}")
plt.show()
"""

