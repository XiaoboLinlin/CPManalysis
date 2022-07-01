import numpy as np
import signac
import os
import mdtraj as md
from CPManalysis.capa import *

# def get_edge_position(universe, left_resid = 1, right_resid=3):
    
#     left = universe.select_atoms('resid {}'.format(left_resid))
#     right = universe.select_atoms('resid {}'.format(right_resid))
#     left_edge_zdistance = max(left.positions[:,2])
#     right_edge_zdistance = max(right.positions[:,2])
    
    

def charge_hist(res_universe, charge, pos_range, direction='z', n_frames = 2500, bins = 100):
    """Calculate charge histogram for atoms in a distance range in a direction
        Those atoms should have fixed positions

    Args:
        res_universe (mda.universe): for the target residue
        charge (npy): _description_
        pos_range (list/np.array): position range for target residue, the atoms in the range will be considered
        direction (str): the direction for the pos_range
        n_frames (_type_): the last number of frames in charge will be considered for histogram
        bins (int, optional): _description_. Defaults to 100.
        
    Returns:
        new_bins, hist_ (np.array): _description_
    """
    
    select_atoms = res_universe.select_atoms('prop {0} > {1} and prop {0} < {2}'.format(direction, pos_range[0], pos_range[1]))
    last_frames_charge = charge[-n_frames:]
    charge_target = last_frames_charge[:, select_atoms.indices]
    charge_target = charge_target.flatten()
    
    hist, bins = np.histogram(charge_target, bins = bins, weights=np.ones(len(charge_target)) / len(charge_target))
    new_bins = bins[1:] - (bins[1]-bins[0])/2
    
    return new_bins, hist


    