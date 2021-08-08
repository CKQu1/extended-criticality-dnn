import math
import torch
import numpy as np

def get_layerWise_norms(net):
    w = []
    g = []
    for p in net.parameters():    
        if p.requires_grad:
            w.append(p.view(-1).norm().detach().cpu().numpy())
            g.append(p.grad.view(-1).norm().detach().cpu().numpy())
    return w, g

def get_grads(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.grad.view(-1))
    grad_flat = torch.cat(res)
    return grad_flat
