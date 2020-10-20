import re
from enum import Enum
from typing import Dict, Union, List

import mne
from mne.utils import warn
from mne_bids.path import (_parse_ext)
from mne_bids.tsv_handler import _from_tsv, _to_tsv

MINIMAL_BIDS_ENTITIES = ('subject', 'session', 'task',
                         'acquisition', 'run', 'datatype')


class ChannelMarkers(Enum):
    """Keyword markers for channels."""

    # non-eeg markers
    NON_EEG_MARKERS = [
        "DC",
        "EKG",
        "REF",
        "EMG",
        "ECG",
        "EVENT",
        "MARK",
        "STI014",
        "STIM",
        "STI",
        "RFC",
    ]
    # bad marker channel names
    BAD_MARKERS = ["$", "DC", "STI"]


class BadChannelDescription(Enum):
    FLAT = 'flat-signal',
    HIGHFREQ = 'high-freq-noise'


def _replace_ext(fname, ext, verbose=False):
    if verbose:
        print(f"Trying to replace {fname} with extension {ext}")

    fname, _ext = _parse_ext(fname, verbose=verbose)
    if not ext.startswith("."):
        ext = "." + ext

    return fname + ext


def _update_sidecar_tsv_byname(
        sidecar_fname: str,
        name: Union[List, str],
        colkey: str,
        val: str,
        allow_fail=False,
        verbose=False,
):
    """Update a sidecar JSON file with a given key/value pair.

    Parameters
    ----------
    sidecar_fname : str
        Full name of the data file
    name : str
        The name of the row in column "name"
    colkey : str
        The lower-case column key in the sidecar TSV file. E.g. "type"
    val : str
        The corresponding value to change to in the sidecar JSON file.
    """
    # convert to lower case and replace keys that are
    colkey = colkey.lower()

    if isinstance(name, list):
        names = name
    else:
        names = [name]

    # load in sidecar tsv file
    sidecar_tsv = _from_tsv(sidecar_fname)

    for name in names:
        # replace certain apostrophe in Windows vs Mac machines
        name = name.replace("’", "'")

        if allow_fail:
            if name not in sidecar_tsv["name"]:
                warn(
                    f"{name} not found in sidecar tsv, {sidecar_fname}. Here are the names: {sidecar_tsv['name']}"
                )
                continue

        # get the row index
        row_index = sidecar_tsv["name"].index(name)

        # write value in if column key already exists,
        # else write "n/a" in and then adjust matching row
        if colkey in sidecar_tsv.keys():
            sidecar_tsv[colkey][row_index] = val
        else:
            sidecar_tsv[colkey] = ["n/a"] * len(sidecar_tsv["name"])
            sidecar_tsv[colkey][row_index] = val

    _to_tsv(sidecar_tsv, sidecar_fname)


def _check_bids_parameters(bids_kwargs: Dict) -> Dict:
    if not all([entity in bids_kwargs for entity in MINIMAL_BIDS_ENTITIES]):
        raise RuntimeError(f'BIDS kwargs parameters are missing an entity. '
                           f'All of {MINIMAL_BIDS_ENTITIES} need to be passed '
                           f'in the dictionary input. You passed in {bids_kwargs}.')

    # construct the entities dictionary
    entities = {entity: bids_kwargs[entity] for entity in MINIMAL_BIDS_ENTITIES}

    return entities


def _look_for_bad_channels(
        ch_names, bad_markers: List[str] = ChannelMarkers.BAD_MARKERS.name, datatype: str = ''
):
    """Looks for hardcoding of what are "bad ch_names".

    Parameters
    ----------
    ch_names : (list) a list of str channel labels
    bad_markers : (list) of string labels

    Returns
    -------
    bad_channels : list
    """
    orig_chdict = {ch.upper(): ch for ch in ch_names}

    ch_names = [c.upper() for c in ch_names]

    # initialize a list to store channel label strings
    bad_channels = []

    # look for ch_names without letter
    bad_channels.extend([ch for ch in ch_names if not re.search("[a-zA-Z]", ch)])

    # scalp EEG has non-numbered electrodes
    if datatype != 'eeg':
        # look for ch_names that only have letters - turn off for NIH pt17
        letter_chans = [ch for ch in ch_names if re.search("[a-zA-Z]", ch)]
        bad_channels.extend([ch for ch in letter_chans if not re.search("[0-9]", ch)])

    # for bad_marker in bad_markers:
    #   bad_channels.extend([ch for ch in ch_names if re.search("[bad_marker]", ch])
    if "$" in bad_markers:
        # look for ch_names with '$'
        bad_channels.extend([ch for ch in ch_names if re.search("[$]", ch)])
    if "DC" in bad_markers:
        badname = "DC"
        bad_channels.extend([ch for ch in ch_names if badname in ch])
    if "STI" in bad_markers:
        badname = "STI"
        bad_channels.extend([ch for ch in ch_names if badname in ch])

    if datatype != 'eeg':
        if "FZ" in bad_markers:
            badname = "FZ"
            bad_channels.extend([ch for ch in ch_names if ch == badname])
        if "GZ" in bad_markers:
            badname = "GZ"
            bad_channels.extend([ch for ch in ch_names if ch == badname])
    # extract non eeg ch_names based on some rules we set
    non_eeg_channels = [
        chan
        for chan in ch_names
        if any([x in chan for x in ChannelMarkers.NON_EEG_MARKERS.value])
    ]
    # get rid of these ch_names == 'e'
    non_eeg_channels.extend([ch for ch in ch_names if ch == "E"])
    non_eeg_channels.extend([ch for ch in ch_names if ch.startswith('$')])
    bad_channels.extend(non_eeg_channels)

    bad_channels = [orig_chdict[ch] for ch in bad_channels]
    return bad_channels


def _channel_text_scrub(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Clean and formats the channel text inside a MNE-Raw data structure.

    Parameters
    ----------
    raw : MNE-raw data structure
    """

    def _reformatchanlabel(label):  # noqa
        """Process a single channel label.

        To make sure it is:

        - upper case
        - removed unnecessary strings (POL, eeg, -ref)
        - removed empty spaces
        """

        # hard coded replacement rules
        # label = str(label).replace("POL ", "").upper()
        label = str(label).replace("POL", "").upper()
        label = label.replace("EEG", "").replace("-REF", "")  # .replace("!","1")

        # replace "Grid" with 'G' label
        label = label.replace("GRID", "G")
        # for BIDS format, you cannot have blank channel name
        if label == "":
            label = "N/A"
        return label

    # apply channel scrubbing
    raw = raw.rename_channels(lambda x: x.upper())

    # encapsulated into a try statement in case there are blank channel names
    # after scrubbing these characters
    try:
        raw = raw.rename_channels(
            lambda x: x.strip(".")
        )  # remove dots from channel names
        raw = raw.rename_channels(
            lambda x: x.strip("-")
        )  # remove dashes from channel names - does this not handle pt11?
    except ValueError as e:
        print(f"Ran into an issue when debugging: {raw.info}")
        raise ValueError(e)

    raw = raw.rename_channels(lambda x: x.replace(" ", ""))
    raw = raw.rename_channels(
        lambda x: x.replace("’", "'")
    )  # remove dashes from channel names
    raw = raw.rename_channels(
        lambda x: x.replace("`", "'")
    )  # remove dashes from channel names
    raw = raw.rename_channels(lambda x: _reformatchanlabel(x))

    return raw
