

import numpy as np

def calc_density_distribution(trj_total, last_n_frame = 2500, res_name = 'tfsi', binwidth = 0.1, axis = 2):
    """calculate com density distribution along one direction

    Args:
        trj_total (md.traj): _description_
        last_n_frame (int or list): _description_. Defaults to 2500. Can select a range by using list type
        res_name (str, optional): _description_. Defaults to 'all'. It can be like 'emim
        binwidth (float, optional): _description_. Defaults to 0.1.
        axis (int): analyze density along which axis. 2 is z axis, 0 is x axis, 1 is y axis

    Returns:
        new_bins, new_hist: _description_
    """
    if res_name == 'all':
        trj = trj_total.atom_slice(trj_total.top.select('all'))
    else:
        trj = trj_total.atom_slice(trj_total.top.select('resname {}'.format(res_name)))
    
    if type(last_n_frame) == list:
        trj = trj[last_n_frame[0]:last_n_frame[1]]
    else:
        trj = trj[-last_n_frame:]
        
    all_xyz = trj.xyz
    total_xyz = all_xyz

    total_xyz = np.reshape(total_xyz, (-1,3))
    data = total_xyz[:, axis]
    # binwidth = 0.1
    box = trj_total.unitcell_lengths[0]
    
    box_length = list(box[:])
    box_length.remove(box[axis])
    bin_volumn = binwidth * box_length[0] * box_length[1]
    n_frame = last_n_frame if type(last_n_frame) is int else int(last_n_frame[1] - last_n_frame[0])
    total_bin_volumn = bin_volumn * n_frame
    # avg_data = data/total_bin_volumn

    # gro_xyz = trj_total[-1].xyz[0]
    # gro_xyz = gro_xyz[:,2]
    
    # bins = np.arange(min(gro_xyz), max(gro_xyz) + binwidth, binwidth)
    bins = np.arange(0, box[axis] + binwidth, binwidth)
    
    hist, bin_edges = np.histogram(data, bins=bins)

    new_hist = hist/total_bin_volumn
    new_bins = bins + (bins[1] - bins[0])/2 
    new_bins = new_bins[:-1]
    return new_bins, new_hist