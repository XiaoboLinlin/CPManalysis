import numpy as np
from scipy import stats
import signac
import os

from sympy import total_degree
project = signac.get_project()

def find_cdc_boundary(one_trj, res_id):
    """the z value for cdc edge boundary

    Args:
        one_frame_trj (_type_): _description_
        res_id (int): if 0, left side cdc edge; if 2, right side cdc edge
        

    Returns:
        z value for boundary (float): unit is nm
    """
    one_frame_trj = one_trj.atom_slice(one_trj.top.select('resid {}'.format(res_id)))
    if res_id == 0:
        max_value = max(one_frame_trj.xyz[0,:,2])
        return max_value
    if res_id == 2:
        min_value = min(one_frame_trj.xyz[0,:,2])
        return min_value
    
def num_density(trj, density=False, axis = 2):
    """calcuate the number of particles or density in the full box range based histrogram

    Args:
        trj (md.traj): can be one frame or multiple frames
        density (bool, optional): _description_. Defaults to False.

    Returns:
        hist: _description_
    """
    total_xyz  = trj.xyz
    total_xyz = np.reshape(total_xyz, (-1,3))
    data = total_xyz[:,axis]
    binwidth = 0.1
    box = trj.unitcell_lengths[0]
    bins = np.arange(0, box[2], binwidth)
    hist, bin_edges = np.histogram(data, bins=bins)
    
    if density:
        bin_volumn = binwidth * box[0] * box[1]
        total_bin_volumn = bin_volumn * len(trj)
        new_hist = hist/total_bin_volumn
        
    new_bins = bins + (bins[1] - bins[0])/2 
    new_bins = new_bins[:-1]
    
    return new_bins, new_hist


def num_density_chunk(trj, bins = [0,3,6], axis = 2):
    """calcuate the number of particles or density in the a customized chunk range

    Args:
        trj (md.traj): can be one frame or multiple frames

    Returns:
        bins, hist: _description_
    """
    total_xyz  = trj.xyz
    total_xyz = np.reshape(total_xyz, (-1,3))
    data = total_xyz[:,axis]
    hist, bin_edges = np.histogram(data, bins=bins)
    return bins, hist


def chunk_mean(trj_com, bins, new_time_space = 0.2):
    """calculte the average number of molecules in a chunk time, and also in a chunk distance

    Args:
        trj_com (_type_): _description_
        bins (_type_): _description_

    Returns:
        bins, mean_value: for example, mean_value[0] is the first bin range of values
    """
    hist_total =[]
    # bins, hist = num_density_chunk(trj_com[0], bins = [0, 5, left_boundary])
    ###calculate each frame histogram using custimized bins
    for i in range(len(trj_com)):
        bins, hist = num_density_chunk(trj_com[i], bins = bins)
        hist_total.append(hist)
    
    hist_total = np.array(hist_total)
    t = np.arange(len(trj_com)) * 0.002 ## 0.002 is timestep
    new_t = np.arange(t[0], t[-1], new_time_space)
    mean_value = []
    
    ### average the valus from several frames to one averaged frame, i determine which bin range
    for i in range(len(bins)-1):
        charge_distribution = stats.binned_statistic(t, hist_total[:,i], 'mean', bins=new_t)
        mean_value.append(charge_distribution.statistic)
    mean_value = np.array(mean_value)
    return new_t[1:], mean_value


def avg_seeds(case, voltage, seeds = [0,1,2,3], normalize = 'subtract'):
    """ make the each ione_data the same shape and make mean of them.

    Args:
        case (_type_): _description_
        voltage (_type_): _description_
        seeds (list, optional): _description_. Defaults to [0,1,2,3].

    Returns:
        mean_total_list: each element inside is a matrix
    """
    total_list = []
    dict_mean_total_list = {}
    # seeds = [0,1,2,3]
    for res_name in ['emim', 'tfsi', 'li', 'acn', 'wat']:
        try:
            for seed in seeds:
                for job in project.find_jobs({"case": case, "voltage": voltage, "seed": seed}):
                        # print('yes')
                    
                    data_file = os.path.join(job.ws, 'ion_exchange_{}.npy'.format(res_name))
                    ione_data = np.load(data_file)
                    if seed == 3:
                        keep_idx = np.where(ione_data[:,0]>4) # keep data larger than 4 ns
                        ione_data = ione_data[keep_idx[0],:]
                    if normalize == 'subtract':
                        normalize_ione_data = ione_data - ione_data[0,:]
                    if normalize == 'divide':
                        normalize_ione_data = ione_data/ione_data[0,:]
                        a = ione_data - ione_data[0,:]
                        normalize_ione_data[:,0] = a[:,0]
                    if normalize == 'original':
                        normalize_ione_data = ione_data
                        a = ione_data - ione_data[0,:]
                        normalize_ione_data[:,0] = a[:,0]
                    total_list.append(normalize_ione_data)
            len_list = [i.shape[0] for i in total_list]
            min_len = min(len_list)
            new_total_list = [i[0:(min_len-1),:] for i in total_list]
            mean_total_list = np.mean(new_total_list, axis =0 )
            dict_mean_total_list[res_name] = mean_total_list
        except:
            continue
        
        
    return dict_mean_total_list


def X_calc(dict_mean_total_list, case, counter_ion, co_ion, axis = 2):
    """calculate ion-exchange machanism parameter X (10.1021/acsnano.9b09648)

    Args:
        dict_mean_total_list (_type_): result returned from avg_seed
        counter_ion (str):
        co_ion (str):
        axis: 2 is left first distance chunk, 4 is right first distance chunk
    return:
        X_value (np.array)
    """
    #### get n0_counter and n0_co from zero potential
    dict_mean_total_list_0 = avg_seeds(case, voltage = 0, seeds = [1,2], normalize ='original')
    n0_counter = np.mean(dict_mean_total_list_0[counter_ion][:, axis])
    n0_co = np.mean(dict_mean_total_list_0[co_ion][:, axis])
    ###
    
    n_counter = dict_mean_total_list[counter_ion][:, axis]
    n_co = dict_mean_total_list[co_ion][:, axis]
    # n0_counter = n_counter[0]
    # n0_co = n_co[0]
    
    n =  n_counter + n_co
    n0 = n[0]
    
    X_value = (n-n0)/((n_counter-n_co) - (n0_counter - n0_co))
    
    t = dict_mean_total_list[counter_ion][:,0] - dict_mean_total_list[counter_ion][0,0]
    
    return t, X_value


    