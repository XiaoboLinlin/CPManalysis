

import numpy as np

def calc_density_distribution(trj_total, last_n_frame = 2500, res_name = 'tfsi', binwidth = 0.1):
    """calculate com density distribution along one direction

    Args:
        trj_total (md.traj): _description_
        last_n_frame (int, optional): _description_. Defaults to 2500.
        res_name (str, optional): _description_. Defaults to 'tfsi'.
        binwidth (float, optional): _description_. Defaults to 0.1.

    Returns:
        new_bins, new_hist: _description_
    """
    
    trj = trj_total.atom_slice(trj_total.top.select('resname {}'.format(res_name)))
    trj = trj[-last_n_frame:]
    all_xyz = trj.xyz
    total_xyz = all_xyz

    total_xyz = np.reshape(total_xyz, (-1,3))
    data = total_xyz[:,2]
    binwidth = 0.1
    box = trj_total.unitcell_lengths[0]
    bin_volumn = binwidth * box[0] * box[1]
    total_bin_volumn = bin_volumn * last_n_frame
    # avg_data = data/total_bin_volumn

    gro_xyz = trj_total[-1].xyz[0]
    gro_xyz = gro_xyz[:,2]
    bins = np.arange(min(gro_xyz), max(gro_xyz) + binwidth, binwidth)

    hist, bin_edges = np.histogram(data, bins=bins)

    new_hist = hist/total_bin_volumn
    new_bins = bins + (bins[1] - bins[0])/2 
    new_bins = new_bins[:-1]
    return new_bins, new_hist