def iter_dict_or_list(data_structure):
    """
    Parameters
    ----------
    data_structure : dict or list

    Returns
    -------
    keys_or_indices :
        If data_structure is a dict, keys_or_indices are the keys.
        If data_structure is a list, keys_or_indices are the indices.
    """

    keys_or_indices = data_structure if isinstance(data_structure, dict) else range(len(data_structure))
    return keys_or_indices
