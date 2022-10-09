## capacitance analysis 
import numpy as np
import signac
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


project = signac.get_project()


def get_potential(case, range='middle', voltage = 0, seeds = [0,1,2,3]):
    # os.chdir('/global/project/projectdirs/m1046/Xiaobo/project/self_project/cdc_clp_2_lm_cori_3more_analysis/src/')
    """get potential at different location range

    Args:
        case (str): the case name
        range (str, optional): 'middle' or 'start' or 'end' location. Defaults to 'middle'.
        voltage (int, optional): _description_. Defaults to 0.
        seeds (list, optional): _description_. Defaults to [0,1,2,3].

    Returns:
        float: the calculated values of potential at different locations
    """
    potential = []
    for seed in seeds:
        for job in project.find_jobs({"case": case, "voltage": voltage, "seed": seed}):
            potential_file = os.path.join(job.workspace(), "potential.npy")
            data = np.load(potential_file)
            # x = data[:,0]/10  
            y = data[:,1] 
            length= len(y)
            # the range is middle 
            if range == 'middle':
                lower_range = int(length/2-length/8)
                upper_range = int(length/2+length/8)
                middle_potential = y[lower_range:upper_range]
                potential_ = np.mean(middle_potential)
            elif range == 'start':
                potential_= y[0]
            elif range == 'end':
                potential_= y[-1]
            potential.append(potential_)
    
    avg_po = np.mean(potential)
    return avg_po

def get_electrode_potential_diff(case, side, voltage, seed):
    """ get the electrode potential (the difference between electrode and electrolyte) 
        relative to the pzc (the difference between electrode and electrolyte)

    Args:
        side (str): 'positive' means positive electrode
        voltage (float): _description_
    Returns:
        float_: the value of the electrode potential relative to the pzc
    """
    if side == 'positive':
        range_ = 'start'
    elif side == 'negative':
        range_ = 'end' 
    start_po = get_potential(case, range= range_, voltage=voltage, seeds=[seed])
    middle_po = get_potential(case, range='middle',voltage=voltage, seeds=[seed])
    electrode_potential = start_po - middle_po
    start_po_pzc = get_potential(case, range= range_, voltage=0)
    middle_po_pzc = get_potential(case, range='middle',voltage=0)
    electrode_potential_pzc = start_po_pzc - middle_po_pzc
    electrode_potential_diff = electrode_potential - electrode_potential_pzc
    return electrode_potential_diff 

def get_electrode_charge(case, voltage, seed, fraction = 1/5):
    """calcualte the charge accumulated in the positive electrode

    Args:
        case (str): _description_
        voltage (float): _description_
        seed (int): _description_
        fraction (_type_, optional): _description_. Defaults to 1/5.

    Returns:
        float: the avg accumulated charge over the last fraction of charge
    """
    for job in project.find_jobs({"case": case, "voltage": voltage, "seed": seed}):
        charge_file = os.path.join(job.workspace(), "pele_charge.npy")
        charge = np.load(charge_file)
        charge =charge[:,1]
        length = len(charge)
        charge_fraction = int(length*fraction)
        charge = charge[-charge_fraction:]
        avg_charge = np.mean(charge)
    return avg_charge

def unit_convert(charge, voltage, atom_mass =  12.011, n_atom=3620):
    """convert e/(g*V) to F/g

    Args:
        charge (float): the value of elementary charge
        voltage (float):
        atom_mass (float): the mass for one atom, unit is amu
        n_atom (int): the number of atoms

    Returns:
        float: the value of capacitance (F/g)
    """
    import unyt as unit
    voltage = voltage * unit.V
    cdc_mass = atom_mass * n_atom * unit.amu
    charge_value = charge * unit.qp
    capa = charge_value / cdc_mass / voltage
    capa = capa.to('F/g')
    return capa

def unit_convert_C_g(charge, atom_mass =  12.011, n_atom=3620):
    """convert e/amu to C/g

    Args:
        charge (float): the value of elementary charge
        atom_mass (float): the mass for one atom, unit is amu
        n_atom (int): the number of atoms

    Returns:
        float: the value of capacitance (C/g)
    """
    import unyt as unit
    charge_value = charge * unit.qp
    cdc_mass = atom_mass * n_atom * unit.amu
    e_g = charge_value/cdc_mass
    e_g = e_g.to('C/g')
    return e_g



def plot_V_charge(case, plot = False):
    """plot the charge vs voltages

    Args:
        case (_type_): _description_
        plot (bool, optional): if you want to plot it out. Defaults to False.

    Returns:
        float: 
            x: electrode potential (V) (potential difference between electrode and bulk relative to the PZC)
            y: electrode charge (e)
    """
    cases = [case]
    voltages = np.arange(0,3+0.25,0.25)
    seeds = [0,1,2,3]
    # seeds = [3]

    plt.rcParams["figure.figsize"] = [5, 4]
    plt.rcParams["figure.autolayout"] = True
    # plt.figure(figsize=(6,5))
    norm = mpl.colors.Normalize(vmin=voltages.min(), vmax=voltages.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    cmap.set_array([])
    if plot:
        fig, ax = plt.subplots(dpi = 100)
    x = []
    y = []
    for case in cases:
        for voltage in voltages:
            for seed in seeds:
                ### 
                p_electrode_diff = get_electrode_potential_diff(case, 'positive', voltage, seed)
                n_electrode_diff = get_electrode_potential_diff(case, 'negative', voltage, seed)
                electrode_charge =  get_electrode_charge(case, voltage, seed, fraction = 1/5)
                electrode_charge = unit_convert_C_g(electrode_charge)
                # print(seed, voltage, p_electrode_diff, p_integral_capa, n_electrode_diff,  n_integral_capa)
                if case == 'neat_emimtfsi':
                    color = 'black'
                if case == 'acn_emimtfsi':
                    color = 'red'
                if case == 'acn_litfsi':
                    color = 'blue'
                if case == 'wat_litfsi':
                    color = 'purple'
                
                x.append(p_electrode_diff)
                x.append(n_electrode_diff)
                y.append(electrode_charge)
                y.append(-electrode_charge)
                
    if plot:
        ax.plot(x,y, 'o', c= color, markersize=3)
        ax.set_ylabel('Charge (C/g)')
        ax.set_xlabel('Voltage (V)')
    return x, y

def plot_gp(x, 
            y, 
            pred_range=[-1.7,1.9], 
            plot = True, 
            length_scale = 1, length_scale_bounds=(1e-5, 1e5), 
            kernel_choice = ['rbf', 'white_kernel','dotproduct'], 
            sigma_0=1, sigma_0_bounds=(1e-5, 1e5)):
    """calcuate Gaussian process regression for x and y; plot or not plot
    Args:
        x (_type_): electrode potential (V) (potential difference between electrode and bulk relative to the PZC)
        y (_type_): electrode charge (C/g)
        plot (boolen): plot out or not
        length_scale (int, optional): for RBF decay. Defaults to 1.
        kernel_choice (list): defaults to ['rbf', 'white_kernel','dotproduct']
        limit (_type_, optional): lower boundary of length_scale_bounds. Defaults to 1e-5.

    Returns:
        np array: 
            x_pred: predicted voltage points after Gaussian process regression, unit (V)
            y_pred: predicted y values after gp 
            sigma: standard deviation associated with y
    """
    if plot:
        fig, ax = plt.subplots(dpi = 150)
    from sklearn import gaussian_process
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, DotProduct, RBF
    # kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    kernel_dict = {
        'rbf': RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds),
        'matern':  Matern(length_scale=2, nu=3/2),
        'white_kernel': WhiteKernel(noise_level=100),
        'dotproduct':DotProduct(sigma_0=sigma_0,sigma_0_bounds=sigma_0_bounds)
    }
    
    for i, item in enumerate(kernel_choice):
        if i == 0:
            kernel =kernel_dict[item]
        else:
            kernel +=kernel_dict[item]
    
    # if kernel_choice == 'rbf':
    #     kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=100) + DotProduct(sigma_0=sigma_0,sigma_0_bounds=sigma_0_bounds)
    # elif kernel_choice == 'rbf_no_dot':
    #     kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=100)
    # elif kernel_choice == 'matern':
    #     kernel = Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=100) #+ DotProduct()
        
    
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    x = np.array(x)
    X = x.reshape(-1, 1)
    gp.fit(X, y)
    print(gp.kernel_)
    
    # if 'dotproduct' not in kernel_choice:
    #     if gp.kernel_.k1.length_scale < 0.1:
    #         print('weird length scale {}, which will be ignored'.format(gp.kernel_.k1.k1.length_scale))
    #         raise ValueError('weird length scale {}, which will be ignored'.format(gp.kernel_.k1.k1.length_scale))
    # else:
    #     if gp.kernel_.k1.k1.length_scale < 0.1:
    #         print('weird length scale {}, which will be ignored'.format(gp.kernel_.k1.k1.length_scale))
    #         raise ValueError('weird length scale {}, which will be ignored'.format(gp.kernel_.k1.k1.length_scale))
    
    x_pred = np.linspace(pred_range[0], pred_range[1]).reshape(-1,1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    x_pred = x_pred.flatten()
    if plot:
        ax.scatter(x, y, label="Observations")
        ax.plot(x_pred, y_pred, label="Mean prediction")
        ax.fill_between(
            x_pred,
            y_pred - 1.96 * sigma,
            y_pred + 1.96 * sigma,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        ax.legend()
        ax.set_xlim(-1.7,2)
        ax.set_xlabel("{}".format("Voltages (V)"))
        ax.set_ylabel("{}".format("Charge density (C/g)"))
    return x_pred, y_pred, sigma