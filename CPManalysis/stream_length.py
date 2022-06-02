    
    

def stream_length(total_list):
    """a list that have many other list inside, we want to stream the list to make 
            each internal list inside the list have the same length.

    Args:
        total_list (list): a list that has multiple internal lists (different length)

    Returns:
        new_total_list (list): a list that has multiple internal lists that have the same length
    """
    len_list = [i.shape[0] for i in total_list]
    min_len = min(len_list)
    new_total_list = [i[0:(min_len-1)] for i in total_list]
    return  new_total_list