### capacitance in a distance range
import CPManalysis.capa as capa
from CPManalysis.charge_dist import charge_dist
import numpy as np
import signac
import os
project = signac.get_project()

def get_gro_file(case):
    for job in project.find_jobs({'case': case, 'seed': 3, 'voltage': 3}):
        gro_file = os.path.join(job.workspace(), "system_lmp.gro")
    return gro_file
        
        
def range_sum(charge, distance, range_distance):
    """calculate the charge in a distance range

    Args:
        charge (_type_): avg_charge_dist from charge_dist()
        distance (_type_): distance from charge_dist()
        range_distance (_type_): like [[1,2],[3,4]]

    Returns:
        list: sum of charge in each distance range
    """
    sum_select_charge_list = []
    for range_ in range_distance:
        range_idx = np.where((distance > range_[0]) & (distance < range_[1]))[0]
        select_charge = charge[range_idx]
        sum_select_charge = np.sum(select_charge)
        sum_select_charge_list.append(sum_select_charge)
    
    return np.array(sum_select_charge_list)

def run_range(total_range, 
              electrode_side = 'positive',
              cases=['neat_emimtfsi', 'acn_emimtfsi', 'acn_litfsi', 'wat_litfsi'],
               voltages =np.arange(0,3+0.25,0.25),
               seeds=[0,1,2,3]):
    """compute the electrode potential, electrode charge, electrode capacitance in a distance range like [[4, 5.25], [5.25, 5.8], [5.8, 6.1]]

    Args:
        total_range (list): The target the distance range like [[4, 5.25], [5.25, 5.8], [5.8, 6.1]]. nm
        electrode_side (str): The electrode you are focusing on
        cases (list, optional): _description_. Defaults to ['neat_emimtfsi', 'acn_emimtfsi', 'acn_litfsi', 'wat_litfsi'].
        voltages (_type_, optional): _description_. Defaults to np.arange(0,3+0.25,0.25).
        seeds (list, optional): _description_. Defaults to [0,1,2,3].

    Returns:
        np.array: electrode potential, charge, capacitance for a distance range
    """
    charge_list = [] ### total charge in a range
    charge_density_list = [] ### charge density in a range (C/g)
    capa_list = []
    electrode_potential = []
    # n_capa_list = []
    for case in cases:
        print('start ', case)
        gro_file = get_gro_file(case)
        for voltage in voltages:
            print('start voltage {} for {}'.format(voltage, case))
            for seed in seeds:
                new_bins, avg_seed_charge_dist = charge_dist(case, voltage, gro_file, seeds = [seed], statistic='sum')
                new_bins_count, avg_seed_count_dist = charge_dist(case, voltage, gro_file, seeds = [seed], statistic='count')
                
                sum_select_array = range_sum(avg_seed_charge_dist, new_bins, total_range)
                n_atoms_array = range_sum(avg_seed_count_dist, new_bins_count, total_range)
                
                electrode_diff = capa.get_electrode_potential_diff(case, electrode_side, voltage, seed)
                # n_electrode_diff = capa.get_electrode_potential_diff(case, 'negative', voltage, seed)
                # integral_capa =  sum_select_array/electrode_diff
                # n_integral_capa =  -sum_select_array/n_electrode_diff
                integral_capa = capa.unit_convert(sum_select_array, electrode_diff, n_atom=n_atoms_array)
                charge_density = capa.unit_convert_C_g(sum_select_array, n_atom=n_atoms_array)
                # n_integral_capa = capa.unit_convert(-sum_select_array, n_electrode_diff, n_atom=n_atoms_array)
                capa_list.append(integral_capa)
                electrode_potential.append(electrode_diff)
                charge_list.append(sum_select_array)
                charge_density_list.append(charge_density)
    return np.array(electrode_potential), np.array(charge_list), np.array(capa_list), np.array(charge_density_list)



    
    