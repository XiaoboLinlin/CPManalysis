import numpy as np
import signac
import os
import warnings
import pandas as pd
from itertools import combinations
project = signac.get_project()

class post_analysis:
    def __init__(self, case_name, 
                 first_n_frame = 12500,
                 idx=1, 
                 seeds = [0,1,2,3],
                 voltages=np.arange(0.25,3+0.25,0.25), 
                 fit_func = 'func_stretch', 
                 combination_num=1, 
                 para_limit=[17, 6, 15], 
                 maxfev = 800):
        """initialize

        Args:
            case_name (str): case name
            idx (int):
                0: the first column
                1: the second column
                .....
            voltages (list or np.array, optional): voltages values. Defaults to np.arange(0.25,3+0.25,0.25).
            fit_func (str): 'func_bi' or 'func_stretch'
                'func_bi': sigma, c, tao1, tao2 (idx = 0, 1, 2, 3)
                'func_stretch': sigma, tao, beta (idx = 0, 1, 2)
            combination_num (int): 
                for reseed, for example, combination_num =3, if seed = [0,1,2,3], then new seed will be like [0,1,2], [0,1,3],[1,2,3]
                Note: if combination_num = 1, it means we use the original seed without reseed. Convienient utilization!
            para_limit (list): any value above this will be ignored.
        """
        self.case = case_name
        self.idx = idx
        self.first_n_frame = first_n_frame
        self.voltages = voltages
        self.fit_func = fit_func
        self.combination_num = combination_num
        self.para_limit=para_limit
        self.maxfev = maxfev
        self.seeds  = seeds
    
    def case_runner(self):
        """for each case, it produce the a dictionary, which contains all voltages, and each voltage has 4 seeds, and each seed has a np array charge (sum of charge for positive electrode)

        Args:
            case_name (str): the name of case

        Returns:
            list: list for all voltages, and each voltage contain 4 seeds of np array charge (over time)
        """
        cases = ['neat_emimtfsi', 'acn_emimtfsi', 'acn_litfsi', 'wat_litfsi']
        
        # voltages=np.array([0.5])
        # seeds = [1]
        i  = 0
        charge_list = []
        # charge_dict={}
        # for v in voltages:
        #     charge_dict[v] = []

        for voltage in self.voltages:
            for case in cases:
                for seed in self.seeds:
                    for job in project.find_jobs():
                        if job.statepoint()['case'] == case and job.statepoint()['seed'] == seed and job.statepoint()['case'] == self.case and job.statepoint()['voltage'] == voltage:
                            # print(job.id)
                            charge_file = os.path.join(job.workspace(), "pele_charge.npy")
                            
                            charge = np.load(charge_file)
                            charge= charge[:self.first_n_frame]
                            # charge_dict[voltage].append(charge[:,1])
                            charge_list.append(charge[:,1])
                            # i = i + 1
                            # print(i)
        return charge_list
    
    def fit_seeds(self, charge_list, seed_num = -1):
        """fit each seed in each voltage
        Args:
            charge_dict (list): 2d list
            seed_num: the number of seeds

        Returns:
            pd dataframe: df that has been cleaned (obviously wrong data are deleted). For example, which tao = 1 M or 20, the correponding seed should be deleted.
        """
        if seed_num == -1:
            seed_num = len(self.seeds)
        from scipy.optimize import curve_fit
        
        def func_bi(t, sig, c, tao1, tao2):
            y = sig*(1 - c * np.exp(-t/tao1) - (1-c)*np.exp(-t/tao2))
            return y

        # def func_stretch(t, sig, tao, beta):
           # # beta = 0.35
        #     y = sig*(1 - np.exp(-(t/tao)**beta)) 
        #     return y
        def func_stretch(t, sig, tao, beta):
            # beta = 1
            y = sig*(1 - np.exp(-(t/tao)**beta)) 
            return y
        
        total_charge_fit = []
        for seed_charge in charge_list:
            t = np.arange(0, len(seed_charge)) * 0.002
            if self.fit_func == 'func_stretch':
                try:
                    popt, pcov = curve_fit(func_stretch, t, seed_charge,maxfev = self.maxfev)
                except:
                    # popt, pcov = curve_fit(func_stretch, t, seed_charge, maxfev = 2000)
                    popt = np.array([np.nan,np.nan,np.nan])
                if popt[0] > self.para_limit[0] or popt[1] > self.para_limit[1] or popt[2] > self.para_limit[2]:
                    popt = np.array([np.nan,np.nan,np.nan])
            else:
                try:
                    popt, pcov = curve_fit(func_bi, t, seed_charge)
                except:
                    popt = np.array([np.nan,np.nan,np.nan,np.nan])
                # if popt[1] > 6 or popt[2] > 15:
                #     popt = np.array([np.nan,np.nan,np.nan])
            total_charge_fit.append([*popt])
        
        voltages = self.voltages
        seeds = np.array(["seed_%d" % i for i in range(0,seed_num)])
        midx = pd.MultiIndex.from_product([voltages, seeds])
        charge_data = pd.DataFrame(total_charge_fit, index = midx)
        charge_data = charge_data.round(2)
        df = charge_data
        df = df.dropna()
        # df.columns = ["sigma", "tao", "beta"] #, 
        # # df.index.names = 'voltage (V)'
        # df.index.names = ['voltage', 'seed']
        return df
    
    def mean_std(self, df, idx = False):
        """ get mean and std for all seeds for target idx in each voltages
        note: self.idx may need to reset 
        Args:
            df (pd frame): _description_

        Returns:
            np.array: total[:,0] is mean, total[:,1] is std, total[:-1] is available seed number
        """
        if idx == False:
            idx = self.idx
        
        self.df = df
        total =[]
        for v in self.voltages:
            each_v = df.loc[v].to_numpy()
            idx_mean = np.mean(each_v[:,idx])
            idx_std = np.std(each_v[:,idx])
            total.append([idx_mean, idx_std, np.shape(each_v)[0]])
            
        total = np.array(total)
        total = np.round(total,2)
        return total
    
    def calculate_mean_std(self):
        """calculate the mean and std for all correct seed in each voltage

        Args:
            idx (int): 
                0: first column
                1: second column (is tao for streched function)
                ....
            # df (pd.dataframe): cleaned dataframe without weird values like too large tao or beta
        return:
            np array: the first column is mean, the second is std, the last one is the number of correct seeds used in the calculation
        """
        print('idx is ', self.idx)
        charge_list = self.case_runner()
        df = self.fit_seeds(charge_list)
        total = self.mean_std(df)
        return total
    
    def transfer_df(self, charge_list):
        """transfer charge_list (list contains all charge for all seeds) to pd.dataframe leveled by voltage and seeds
        important step to make dataframe tidy 
        note: using 4 seeds as default
        Args:
            charge_list (list): _description_
        Returns:
            pd dataframe: well organized df (from 0 level 2d list)
        """
        set_seed_num=len(self.seeds)
        voltages = self.voltages
        print('The number of seeds is in transfer_df()', set_seed_num)
        seeds = np.array(["seed_%d" % i for i in range(0,set_seed_num)])
        midx = pd.MultiIndex.from_product([voltages, seeds])
        charge_data = pd.DataFrame(charge_list, index = midx)
        df_original_charge= charge_data.dropna(axis =1)
        return df_original_charge
    
    def make_new_each_v(self, each_v, num):
        """make new charge array by averaging 2 seeds or 3 seeds. For example, seed = [0,1,2,3], new seeds will be the average from [0,1],[0,2],[0,3]...[2,3], total 6
            Each seed is a charge array
        Args:
            each_v (np.array): charge for all seeds
            num (int): how many seeds you want to use to get the averaged new one seed. 
        """
        
        comb = combinations(self.seeds, num)
        new_each_v = []
        for i in comb:
            each_v_mean = np.mean(each_v[i,:], axis = 0)
            new_each_v.append(each_v_mean)
        return new_each_v
    
    def re_seed(self, charge_list):
        """ make new seeds based on average of different combination. 
        For example, seed = [0,1,2,3], new seeds will be the average from [0,1],[0,2],[0,3]...[2,3], total 6
        note: the combination_num is self.combination_num
        
        Args:
            charge_list (np.array): see above

        Returns:
            np.array: new seeded np.array
        """
        
        voltages = self.voltages
        df_original_charge= self.transfer_df(charge_list)
        comb = combinations(self.seeds, self.combination_num)
        self.current_number_reseed = len(list(comb))
        print('the number of reseeds is ', self.current_number_reseed)
        total_new_charge = [] 
        for v in voltages:
            each_v = df_original_charge.loc[v].to_numpy()
            new_each_v = self.make_new_each_v(each_v, self.combination_num)
            # total_new_charge = np.append(total_new_charge, new_each_v)
            total_new_charge.append(new_each_v)
        total_new_charge= np.array(total_new_charge)
        shape = np.shape(total_new_charge)
        total_new_charge_reshape = np.reshape(total_new_charge, (shape[0]*shape[1], shape[2]))
        #total_new_charge is two levels; total_new_charge_reshape has only one level (flattened)
        
        return total_new_charge, total_new_charge_reshape
    
    def reseed_calculate_mean_std(self):
        """
        calculate mean and std for fitted reseed parameters.
        The same as calculate_mean_std(), but for reseed, which means new number of averaged seed for each voltage
        
        return pd dataframe (well organized with different level)
        """
        print('idx is ', self.idx)
        charge_list = self.case_runner()
        total_new_charge, total_new_charge1 = self.re_seed(charge_list)
        new_seed_num = len(list(combinations(self.seeds, self.combination_num)))
        print('new number of seeds is ', new_seed_num)
        df = self.fit_seeds(total_new_charge1, seed_num = new_seed_num)
        total = self.mean_std(df)
        return total
    
    
    def avg_charge_original(self):
        """get avg charge for each voltages, so all seeds in each voltage will be averaged

        Returns:
            _type_: _description_
        """
        charge_list = self.case_runner()
        df_original_charge= self.transfer_df(charge_list)
        avg_charge_total = []
        for v in self.voltages:
            each_v = df_original_charge.loc[v].to_numpy()
            avg_charge_each_v = np.mean(each_v, axis=0)
            avg_charge_total.append(avg_charge_each_v)
        return avg_charge_total

    def run_slice_column_mean(self, fraction):
        """ get the [-th_frame:] charge, and then reseed it, and then get one mean charge for each seed
        
        Args:
            fraction (float): for example, th_frame = 1/5, so the last 1/5 fraction of column number will be kept
            
        returns: 
            total (np.array): total[:,0] is mean, total[:,1] is std, total[:-1] is available seed number
        
        """
        
        all_v_charge = self.case_runner()
        df_all_charge= self.transfer_df(all_v_charge)
        df_all_charge1 = df_all_charge.to_numpy()
        shape = np.shape(df_all_charge1)
        last_fifth = int(shape[1]*fraction)
        df_all_charge2 = df_all_charge1[:, -last_fifth:]
        total_new_charge, total_new_charge_reshape = self.re_seed(df_all_charge2)
        df_all_charge3 = np.mean(total_new_charge_reshape, axis = 1)
        df_all_charge3 = self.transfer_df(df_all_charge3)
        total = self.mean_std(df_all_charge3, idx = 0)
        return total
    
        
        
    

        
    
    