import numpy as np
from scipy import stats
import signac
import os
# from CPManalysis.stream_length import stream_length
import mdtraj as md
project = signac.get_project()

def charge_dist(case, voltage, gro_file, charge_file_name='charge.npy', target_res = 'cdc', seeds = [0,1,2,3], n_frames = 2500, statistic='mean', binwidth = 0.1):
    """calculate the charge distribution in one direction for targeted atoms
        currently, this function only supperts atoms with fixed positions.

    Args:
        case (_type_): _description_
        voltage (float): _description_
        target_res (str): targeted residue to get charge distribution
        seeds (list, optional): _description_. Defaults to [0,1,2,3].
        charge_file_name:
        n_frames : the number of last frames that used for getting averaged charge

    Returns:
        np.array, np.array: new_bins, targeted charge ditribution
    """
    charge_dist = []
    new_bins = []
    for seed in seeds:
        for job in project.find_jobs({"case": case, "voltage": voltage, "seed": seed}):
            charge_file = os.path.join(job.workspace(), charge_file_name)
            charge = np.load(charge_file)
            one_frame = md.load(gro_file)
            target_top_idx = one_frame.top.select('resname {}'.format(target_res))
            target = one_frame.atom_slice(target_top_idx)
            z_data = target.xyz[0,:,2]
            avg_charge = np.mean(charge[-n_frames:], axis = 0)
            
            bins = np.arange(0, max(z_data) + binwidth, binwidth)
            targe_avg_charge = avg_charge[target_top_idx]
            charge_distribution = stats.binned_statistic(z_data, targe_avg_charge, statistic=statistic, bins=bins)
            charge_dist.append(charge_distribution.statistic)
            new_bins.append(bins)
            
    # streamed_charge_dist = stream_length(charge_dist)
    avg_charge_dist = np.mean(charge_dist, axis =0)
    # streamed_bins_dist = stream_length(bins)
    new_bins0= new_bins[0] ## only need the shortest bins one
    distance = new_bins0[1:] - (new_bins0[1]-new_bins0[0])/2
    return np.array(distance), np.array(avg_charge_dist)


