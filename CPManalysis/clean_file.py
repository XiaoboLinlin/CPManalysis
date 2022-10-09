import numpy as np

def del_repeat(x):
    """ delete repeating groups. used in clean_dumpfile function
    For example, [[9,2],[10,2],[13,4],[12,4],[15,4]]
    I will only keep [[9,2],[10,2],[13,4],[15,4]], which is the first and the last of repeating groups
    Args:
        x (list): the list(2d) you want to modify
    """
    import itertools
    a = [list(group) for key, group in itertools.groupby(x, key=lambda v: v[1])]
    new_a =[]
    for i in a:
        new_a.append(i[0])
        new_a.append(i[-1])
    return np.array(new_a)

def del_decrease(x):
    """delete the elements that are right before suddenly decreased elements. For example, the x[:,1] is 
    np.array([2, 3, 5, 2, 5, 5]) : so the row with the first 3 and 5 will be deleted.
    For example, in find_idx, the second column is timestep, which should be increased not decreased

    Args:
        x (np.array, 2d): the array that you want to modify
    """
    x_diff = np.diff(x[:, 1])
    idx = np.where(x_diff<0)[0]
    while idx.size != 0:
        x_diff = np.diff(x[:, 1])
        idx = np.where(x_diff<0)[0]
        x= np.delete(x, idx, axis = 0)
    return x

def clean_dumpfile(file_name, stop_at = -1):
    """clean dump file from lammps, like delete repeated timestep and delete buggy timestep due to sudden break
    Note that: the last frame is ingored

    Args:
        file_name (str): the name of original dump file
        stop_at (int): the step that stop at (default -1 means all step will be used). for example, stop_at = 14050000, we will use all steps before 14050000 (not include 14050000)
    """
    import re
    with open(file_name, mode='r') as f:
        content = f.read()
        
    if stop_at > 0:
        stop_step = re.finditer("ITEM: TIMESTEP\n{}\n".format(stop_at), content)
        stop_find_idx = [[m.start(), m.group().split()[-1]] for m in stop_step]
        content = content[0:stop_find_idx[0][0]]
        
    time_step = re.findall("ITEM: TIMESTEP\n\d*\n", content)
    ts = np.array([i.split()[-1] for i in time_step])
    ts = ts.astype(np.int64)
    diff = np.diff(ts)
    diff_idx = np.where(diff<=0)
    diff_idx = diff_idx[0] + 1
    target_ts = ts[diff_idx]

    ### the last time step index
    find_idx_ = re.finditer('ITEM: TIMESTEP\n{}\n'.format(ts[-1]), content)
    last_ts_idx = [m.start() for m in find_idx_]
    last_ts_idx = last_ts_idx[0]

    regex = ""
    for i, ts_ in enumerate(target_ts):
        if i == 0:
            regex += 'ITEM: TIMESTEP\n{}\n'.format(ts_)
        else:
            regex += '|ITEM: TIMESTEP\n{}\n'.format(ts_)
    # regex
    pat = re.compile(r"{}".format(regex), re.UNICODE)
    find_idx = re.finditer(pat, content)

    find_idx = [[m.start(), m.group().split()[-1]] for m in find_idx] ## [[idx, timestep]]
    find_idx = np.array(find_idx)
    find_idx = find_idx.astype(np.int64)
    ## note: in find_idx, all repeating timestep is counted.
    
    ### delete suddenly decreased elements
    find_idx = del_decrease(find_idx)

    ### delete repeating elements, 
    str_idx_ = del_repeat(find_idx) #str_idx structure is similar to find_idx in this step
    str_idx = str_idx_[:,0] # only keep idx
    ### the range in between will be kept, for example, [1,8], so [0:1] and [8:] will be kept
    str_idx = np.insert(str_idx, 0, 0)
    str_idx = np.append(str_idx, -1)
    str_idx = np.reshape(str_idx, (-1, 2))

    ### make new content with desired range
    length = len(str_idx)
    new_content = ""
    for i, idx in enumerate(str_idx):
        if i == (length-1):
            new_content += content[idx[0]:last_ts_idx]
        else:
            new_content += content[idx[0]:idx[1]]
    return new_content