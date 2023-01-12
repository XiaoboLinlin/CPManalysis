import numpy as np

def profile_reader(file, n_bins):
    """read potential files produced from atc packages

    Args:
        file (str): path to file
         n_bins (int): the number of bins for the potential, 241 for cdc project
    Returns:
        _type_: _description_
    """
    with open(file,"r") as fi:
        lines = []
        for ln in fi:
            if ln.startswith(" "):
                lines.append(ln)
    data = np.array(lines)  
    df = np.loadtxt(data)
    # print(df)
    df = np.split(df, len(df)/n_bins)
    df = np.mean(df, axis = 0)
    return df

def q_np(lmp_trj, n_atom):
    # import sys
    """extract charge data from .lammpstrj to numpy array

    Args:
        lmp_trj (_type_): .lammpstrj
        n_atom (int): the total number of atoms in the system

    Returns:
        np array: charge data in 2d array [n_frames, n_atoms]
    """
    # fin = open(lmp_trj, "r")
    # fin.close()
    # linelist = fin.readlines()
    q_total = []
    with open(lmp_trj, 'r') as f:
        look = False
        i = 0 
        for line in f:
            i += 1 
            if (len(line.split()) >= 7 and line.split()[0] != 'ITEM:') or look==True:
                try:
                    q = float(line.split()[-1])
                except:
                    look = True
                    print('overlaped line here, cleaned up')
                    print(line.split())
                    # q = float(line.split()[-2][:-5])
                    print('line number ', i)
                    break
                q_total.append(q)
    charge = np.array(q_total)
    charge_2d = np.reshape(charge, (int(len(charge)/n_atom),n_atom))
    return charge_2d