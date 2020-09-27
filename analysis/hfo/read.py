import collections
from pathlib import Path

import numpy as np
import scipy.io
from mne_bids import BIDSPath

def _convert_event_to_dict(event_arr, ch_names, sfreq):
    event_dict = collections.defaultdict(list)
    for irow in range(event_arr.shape[0]):
        row = event_arr[irow, :]

        # matlab indices start at 1
        ch_name = ch_names[int(row[0]) -1]

        # convert start/end time to samples inside the data file
        start_sec, end_sec = row[1], row[2]
        event_dict[ch_name].append([start_sec * sfreq, end_sec * sfreq])
    return event_dict


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict


def _tolist(ndarray):
    """
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    """
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def read_hfo_data(fname):
    data_dict = loadmat(fname).get('data')

    print(data_dict.keys())
    # extract ripples, fastripples and intersection
    # as T x 3 array = (channel index, start time second, end time second)
    ripple = np.array(data_dict['R'])
    fast_ripple = np.array(data_dict['FR'])
    frandr_ripple = np.array(data_dict['FRandR'])

    # extract computation time
    compute_time = data_dict['computationtime']

    # sfreq
    sfreq = data_dict['fs']

    # extract ch_names - make sure upper case
    ch_names = [ch.upper() for ch in data_dict['label']]

    # convert each of the HFO defintions into a dictionary
    # of channels and their corresponding endpoints (start_time sample, end_time sample)
    ripple_dict = _convert_event_to_dict(ripple, ch_names, sfreq)
    fastripple_dict = _convert_event_to_dict(fast_ripple, ch_names, sfreq)
    frandr_dict = _convert_event_to_dict(frandr_ripple, ch_names, sfreq)

    hfo_dict = {
        'ripple': ripple_dict,
        'fastripple': fastripple_dict,
        'fastripple_and_ripple': frandr_dict,
        'sfreq': sfreq,
        'compute_time': compute_time
    }
    return hfo_dict

def _get_bids_path(fname):
    subject = fname.split('_')[0]
    run = fname.split('_')[-1].split('.')[0]

    # all presurgery
    session = 'presurgery'

    # these are all interictal datasets
    task = 'interictal'
    if run.isnumeric():
        pass
    else:
        task = f'interictal{run}'

    bids_path = BIDSPath(subject=subject, session=session, task=task,
                         run=run)

if __name__ == '__main__':
    deriv_root = Path('/home/adam2392/hdd/Dropbox/hfo_data/hfo_mat_results/mat')

    for fpath in deriv_root.glob('*.mat'):
        hfo_dict = read_hfo_data(fpath)

        print(hfo_dict.keys())
        break
