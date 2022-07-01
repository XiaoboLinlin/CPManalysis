# import CPManalysis.capa as capa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import CPManalysis.capa as capa
import multiprocessing

class diff_capa:
    def __init__(self, 
                 case = 'acn_emimtfsi',
                 kernel_choice = ['rbf', 'white_kernel','dotproduct'],
                 re_sample='total',
                 fraction_samples=1,
                 n_iterations = 10,
                 replace = True,
                 n_seeds = 4):
        """_summary_

        Args:
            case (): _description_
            kernel_choice (list, optional): _description_. Defaults to ['rbf', 'white_kernel','dotproduct'].
            re_sample (str, optional): _description_. Defaults to 'total'.
            fraction_samples (int, optional): _description_. Defaults to 1.
            n_iterations (int, optional): _description_. Defaults to 10.
            replace (bool, optional): _description_. Defaults to True.
            n_seeds (int, optional): _description_. Defaults to 4.
        """
        self.case = case
        self.kernel_choice = kernel_choice
        self.re_sample=re_sample
        self.fraction_samples=fraction_samples
        self.n_iterations = n_iterations
        self.replace = replace
        self.n_seeds = n_seeds
        
    def run_case(self):
        """to get the original x, y (voltage, charges)

        Returns:
            _type_: _description_
        """
        # case = ['neat_emimtfsi', 'acn_emimtfsi', 'acn_litfsi', 'wat_litfsi']
        # plt.rcParams['figure.dpi'] = 100
        #fig, ax = plt.subplots(dpi = 100)
        # for case in self.case:
        print(self.case)
        x, y = capa.plot_V_charge(self.case)
        return np.array(x), np.array(y)
    
    def run_gp(self,x, y):
        """from original x, y (voltage, charge) to get the fitted gp result

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_pred, y_pred, sigma = capa.plot_gp(x, y, pred_range=[-1.7,1.9], plot=False, length_scale = 2, length_scale_bounds=(0.05, 1e5), kernel_choice=self.kernel_choice)
            # ax.set_xlim(-1.8,2.1)
        gradient_y = np.gradient(y_pred, x_pred)
        return np.array(x_pred), np.array(gradient_y)
    
    def resample_seeds(self, x, y):
        """resample all seeds for each voltage
        Args:
            x (np.array): original voltage (not grouped)
            y (np.array): original charge (not grouped)
            n_iterations (int, optional): each interation resamples all seeds in each voltage. Defaults to 100.
            n_seeds (int, optional): _description_. Defaults to 4.

        Returns:
            np.array: shape is (n_iterations, the number of voltages including seeds, 2), the "2" is for voltage column and charge column
        """
        
        
        if self.re_sample == 'seeds':
            ### group all seeds at each voltage 
            a = x.reshape(-1,self.n_seeds)
            b = y.reshape(-1,self.n_seeds)
            ### make data structure for resample_data (next step)
            c = [np.dstack((a[i], b[i]))[0] for i in range(len(a))]
            
            resample_data_total = []
            for a in range(self.n_iterations):
                resample_data = [self.resample(i, replace=self.replace, n_samples=int(len(i)*self.fraction_samples)) for i in c]
                resample_data_total.append(resample_data)
                
            resample_data_total = np.array(resample_data_total)
            ### reshape it, so all seeds not grouped
            resample_data_total = resample_data_total.reshape(self.n_iterations, -1, 2)
            
        else:
            resample_data_total = []
            data = np.dstack((x,y))[0]
            for a in range(self.n_iterations):
                resample_data = resample(data, replace=self.replace, n_samples=int(len(data)*self.fraction_samples))
                resample_data_total.append(resample_data)
        return np.array(resample_data_total)

    def run_diff_bootstrap(self):
        """_summary_

        Returns:
            _type_: get all x, y for gp results for all iteractions, [:, :, :] -> [n_iterations, x, y]
        """
        
        x, y = self.run_case()
        # a = np.array([x,y])
        # data = a.T
        # rng = np.random.default_rng()
        # res = bootstrap((x, y), run_gp, vectorized=False, paired=True, random_state=rng,n_resamples =3)

        ### x[0::2] is for positive, and x[1::2] is for negative
        clean_x = np.concatenate((x[0::2],x[1::2]))
        clean_y = np.concatenate((y[0::2],y[1::2]))
        resample_data_total = self.resample_seeds(clean_x , clean_y)
        resample_data_total.shape

        new_gp = []
        # new_sample_list = []
        
        
        # ### do parallel computing
        # def multi_iter(value):
        #     x_pred, y_grad = self.run_gp(resample_data_total[i,:,0], resample_data_total[i,:,1])
        #     gp = np.array([x_pred, y_grad]).T
        #     return gp
        #     # new_gp.append(gp)
        #     # plt.plot(gp[:,0], gp[:,1])
        #     # plt.xlabel('Voltage (V)')
        #     # plt.ylabel('Differential capacitance (F/g)')
        
        # pool_obj = multiprocessing.Pool()
        # result = pool_obj.map(multi_iter, range(self.n_iterations))
        # new_gp = np.array(result)
        
        
        for i in range(self.n_iterations):
            # new_sample = resample(data, replace=True, n_samples=len(data))
            # new_sample_list.append(new_sample)
            try:
                x_pred, y_grad = self.run_gp(resample_data_total[i,:,0], resample_data_total[i,:,1])
            except ValueError as err:
                continue
            gp = np.array([x_pred, y_grad]).T
            new_gp.append(gp)
            plt.plot(gp[:,0], gp[:,1])
            plt.xlabel('Voltage (V)')
            plt.ylabel('Differential capacitance (F/g)')
        return np.array(new_gp)
    
    def plot_diff_capa(self, new_gp, plot_choice = ['all','interval']):
        """plot diff capacitance with error band

        Args:
            new_gp (_type_): _description_
            plot_choice (list, optional): _description_. Defaults to ['all','interval'].
        """
        # case_color = {
        #         'neat_emimtfsi':'red',
        #         'acn_emimtfsi':'blue',
        #         'acn_litfsi':'orange',
        #         'wat_litfsi':'black'
        #     }
        if 'all' in plot_choice:
            for i in range(len(new_gp)):
                gp = new_gp[i]
                plt.plot(gp[:,0], gp[:,1], 'o', color = 'blue', alpha=0.1)
                plt.xlabel('Voltage (V)')
                plt.ylabel('Differential capacitance (F/g)')
        
        if 'interval' in plot_choice:
            charge_data = new_gp[:,:,1]
            charge_data_std = np.std(charge_data, axis = 0)
            charge_data_mean = np.mean(charge_data, axis = 0)

            gp_0 = new_gp[0]
            # plt.errorbar(gp_0[:,0], charge_data_mean, yerr=charge_data_std)

            plt.fill_between(gp_0[:,0], charge_data_mean-charge_data_std, charge_data_mean+charge_data_std, facecolor ='orange', alpha=0.5)
            plt.xlabel('Voltage (V)')
            plt.ylabel('Differential capacitance (F/g)')


    