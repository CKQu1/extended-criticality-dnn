import argparse
import heapq
import json
import numpy as np
import os
import pandas as pd
from ast import literal_eval
from os.path import join, normpath, isdir, isfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# -------------------- Path utils --------------------

def njoin(*args):
    return normpath(join(*args))

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2ls(s):
    if isinstance(s, list):
        return s
    elif isinstance(s, str):
        if ',' in s:
            return s.split(',')
        else: 
            return [s]

def find_subdirs(root_dir, matching_str):
    matches = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if matching_str in dirname.lower() and dirpath not in matches:
                matches.append(dirpath)
    return matches

def get_seed(dir, *args):  # for enumerating each seed of training
    #global start, end, seeds, s_part

    if isdir(dir):
        seeds = []
        dirnames = next(os.walk(dir))[1]
        if len(dirnames) > 0:
            for dirname in dirnames:        
                #is_append = (len(os.listdir(njoin(dir, dirname))) > 0)  # make sure file is non-empty
                is_append = True
                for s in args:
                    is_append = is_append and (s in dirname)
                #print(f'{dirname} is_append: {is_append}')  # delete
                if is_append:  
                    #try:        
                    #for s_part in dirname.split(s):
                    assert "model=" in dirname, f'str model= not in {dirname}'
                    start = dirname.find("model=")
                    seeds.append(int(dirname[start+6:]))
                    #except:
                    #    pass       
            #print(seeds)  # delete
            return max(seeds) + 1 if len(seeds) > 0 else 0
        else:
            return 0
    else:
        return 0


def point_to_path(seeds_root, alpha100, g100, seed):

    alpha100, g100, seed = int(alpha100), int(g100), int(seed)
    
    for subdir in os.listdir(seeds_root):
        if f'seed={seed}' in subdir:
            seed_root = njoin(seeds_root, subdir)
            for model_dir in os.listdir(seed_root):
                if f'_{alpha100}_{g100}_{seed}_' in model_dir:
                    return njoin(seed_root, model_dir)
    return None


# -------------------- Main utils --------------------  

def convert_dict(dct):  # change elements of dict its value is a dict
    for key in list(dct.keys()):
        val = dct[key]
        if isinstance(val, dict):
            val_ls = list(val.values())
            assert len(val_ls) == 1, 'val_ls has len greater than one'
            dct[key] = val_ls[0]
    return dct

# -------------------- Others --------------------        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self    

def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value