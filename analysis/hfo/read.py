import scipy.io
from scipy.io import loadmat
import collections

def _convert_event_to_dict(event_arr, ch_names, sfreq):
    event_dict = collections.defaultdict(list)
    for irow in range(event_arr.shape[0]):
        row = event_arr[irow, :]
        ch_name = ch_names[row[0]]

        start_sec, end_sec = row[1], row[2]
        event_dict[ch_name].append([start_sec * sfreq, end_sec *sfreq])
    return event_dict

def read_hfo_data(fname):
    data_dict = loadmat(fname)

    # extract ripples, fastripples and intersection
    # as T x 3 array = (channel index, start time second, end time second)
    ripple = data_dict['R']
    fast_ripple = data_dict['FR']
    frandr_ripple = data_dict['FRandR']

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