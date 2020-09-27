"""API for converting files to BIDS format."""
import json
import os
import random
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Union, List
import collections
import mne
from mne.io import read_raw_edf
from mne.utils import warn
from mne_bids import write_raw_bids, read_raw_bids, get_anonymization_daysback
from mne_bids.path import (BIDSPath, get_entities_from_fname,
                           _find_matching_sidecar, _parse_ext)
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
    BAD_MARKERS = ["$", "FZ", "GZ", "DC", "STI"]


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
        ch_names, bad_markers: List[str] = ChannelMarkers.BAD_MARKERS.name
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
    # look for ch_names that only have letters - turn off for NIH pt17
    letter_chans = [ch for ch in ch_names if re.search("[a-zA-Z]", ch)]
    bad_channels.extend([ch for ch in letter_chans if not re.search("[0-9]", ch)])

    # for bad_marker in bad_markers:
    #   bad_channels.extend([ch for ch in ch_names if re.search("[bad_marker]", ch])
    if "$" in bad_markers:
        # look for ch_names with '$'
        bad_channels.extend([ch for ch in ch_names if re.search("[$]", ch)])
    if "FZ" in bad_markers:
        badname = "FZ"
        bad_channels.extend([ch for ch in ch_names if ch == badname])
    if "GZ" in bad_markers:
        badname = "GZ"
        bad_channels.extend([ch for ch in ch_names if ch == badname])
    if "DC" in bad_markers:
        badname = "DC"
        bad_channels.extend([ch for ch in ch_names if badname in ch])
    if "STI" in bad_markers:
        badname = "STI"
        bad_channels.extend([ch for ch in ch_names if badname in ch])

    # extract non eeg ch_names based on some rules we set
    non_eeg_channels = [
        chan
        for chan in ch_names
        if any([x in chan for x in ChannelMarkers.NON_EEG_MARKERS.value])
    ]
    # get rid of these ch_names == 'e'
    non_eeg_channels.extend([ch for ch in ch_names if ch == "E"])
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


def append_original_fname_to_scans(
        orig_fname: str,
        bids_root: Union[str, Path],
        bids_fname: str,
        overwrite: bool = True,
        verbose: bool = True,
):
    """Append the original filename to *scans.tsv in BIDS data structure.

    Parameters
    ----------
    orig_fname : str
        The original base filename that will be added into the
        'original_filename' columnn.
    bids_root : str | Path
        The root to the BIDS dataset.
    bids_fname : str | BIDSPath
        The BIDS filename of the BIDSified dataset. This should
        correspond to a specific 'filename' in the *scans.tsv file.
    overwrite : bool
        Whether or not to overwrite the row.
    verbose : bool
    """
    # only keep the original filename w/o the extension
    params = get_entities_from_fname(bids_fname)
    bids_path = BIDSPath(**params, root=bids_root)

    # find scans.tsv sidecar
    scans_fpath = _find_matching_sidecar(
        bids_path, suffix="scans.tsv", allow_fail=False
    )
    scans_tsv = _from_tsv(scans_fpath)

    # new filenames
    filenames = scans_tsv["filename"]
    ind = [i for i, fname in enumerate(filenames) if bids_fname in fname]

    if len(ind) > 1:  # pragma: no cover
        raise RuntimeError(
            "This should not happen. All scans should "
            "be uniquely identifiable from scans.tsv file. "
            "The current scans file has these filenames: "
            f"{filenames}."
        )
    if len(ind) == 0:
        raise RuntimeError(
            f"No filename, {bids_fname} found. "
            f"Scans.tsv has these files: {filenames}."
        )

    # write scans.json
    scans_json_path = _replace_ext(scans_fpath, "json")
    scans_json = {
        "original_filename": "The original filename of the converted BIDs dataset. "
                             "Provides possibly ictal/interictal, asleep/awake and "
                             "clinical seizure grouping (i.e. SZ2PG, etc.)."
    }
    with open(scans_json_path, "w") as fout:
        json.dump(scans_json, fout, indent=4)

    # write in original filename
    if "original_filename" not in scans_tsv.keys():
        scans_tsv["original_filename"] = ["n/a"] * len(filenames)
    if scans_tsv["original_filename"][ind[0]] == "n/a" or overwrite:
        scans_tsv["original_filename"][ind[0]] = orig_fname
    else:
        raise RuntimeError(
            "Original filename has already been written here. "
            f"Skipping for {bids_fname}. It is written as "
            f"{scans_tsv['original_filename'][ind[0]]}."
        )
        return

    # write the scans out
    _to_tsv(scans_tsv, scans_fpath)


def _modify_raw_with_json(raw: mne.io.BaseRaw,
                          ch_metadata: Dict, verbose: bool = True) -> mne.io.BaseRaw:
    # get the bad channels and set them
    bad_chs = ch_metadata['bad_contacts']

    # set channel types
    ch_types = ch_metadata['ch_types']
    raw.set_channel_types(ch_types, verbose=verbose)

    # set additional bad channels
    raw.info["bads"].extend(bad_chs)

    return raw


def _set_ch_types(raw, subject, session):
    raw.set_channel_types({ch: 'misc' for ch in raw.ch_names})

    if subject == 'E1':
        grids = [f'C{idx}' for idx in range(1, 65)]
        depths = [f'SAD{idx}' for idx in range(1, 7)] + \
                 [f'SPD{idx}' for idx in range(1, 7)] + \
                 [f'MPOD{idx}' for idx in range(1, 7)] + \
                 [f'IAD{idx}' for idx in range(1, 7)] + \
                 [f'MAOD{idx}' for idx in range(1, 7)] + \
                 [f'IPD{idx}' for idx in range(1, 7)]
        resected = ['C19', 'C20', 'C27', 'C28']
        bads = ['C15'] + [f'C{idx}' for idx in range(17, 21)] + \
               [f'C{idx}' for idx in range(24, 29)] + \
               [f'C{idx}' for idx in range(34, 37)] + \
               ['C51', 'C57', 'C59']
        bads = {
            'C15': BadChannelDescription.HIGHFREQ,
            # 'C55': BadChannelDescription.FLAT,
            # 'C56': BadChannelDescription.FLAT,
            'C57': BadChannelDescription.HIGHFREQ,
            # 'C63': BadChannelDescription.FLAT,
            # 'C64': BadChannelDescription.FLAT,
        }

        # add bad resected channels
        if session in ['postsurgery', 'intraoperative']:
            bad_chs = []
            bad_chs.extend(['C15', 'C17', 'C18', 'C19', 'C20'] + \
                           [f'C{idx}' for idx in range(25, 29)] + \
                           ['C34', 'C51', 'C57', 'C59'])
            bad_chs.extend(depths)
            for ch in bad_chs:
                bads[ch] = BadChannelDescription.FLAT

    elif subject == 'E2':
        grids = [f'C{idx}' for idx in range(1, 65)]
        depths = []
        resected = ['C1', 'C2']
        bads = [f'C{idx}' for idx in range(1, 11)]
        bads = []
    elif subject == 'E3':
        grids = [f'C{idx}' for idx in range(1, 65)]
        depths = \
            [f'ASD{idx}' for idx in range(1, 7)] + \
            [f'PD{idx}' for idx in
             range(1, 7)]  # + [f'PS{idx}' for idx in range(1, 5)] + # [f'AID{idx}' for idx in range(1, 7)] + \
        resected = ['C3', 'C4', 'C11', 'C12', 'C19',
                    'C20', 'C27', 'C28']
        bads = [f'C{idx}' for idx in range(1, 6)] + \
               [f'C{idx}' for idx in range(9, 15)] + \
               [f'C{idx}' for idx in range(17, 24)] + \
               [f'C{idx}' for idx in range(25, 32)] + ['C51']

        bads = {
            'C51': BadChannelDescription.FLAT,
        }
    elif subject == 'E4':
        grids = [f'C{idx}' for idx in range(1, 49)]
        depths = [f'F2AL{idx}' for idx in range(1, 7)] + \
                 [f'F2BC{idx}' for idx in range(1, 7)] + \
                 [f'F2CL{idx}' for idx in range(1, 7)]
        resected = ['C19', 'C20', 'C27', 'C28', 'C35', 'C36']
        bads = [f'C{idx}' for idx in range(3, 6)] + \
               [f'C{idx}' for idx in range(11, 14)] + \
               ['C19', 'C20', 'C27', 'C28', 'C35', 'C36',
                'C47']

        bads = {
            'C1': BadChannelDescription.FLAT,
            'C2': BadChannelDescription.FLAT,
            'C3': BadChannelDescription.HIGHFREQ,
            'C9': BadChannelDescription.FLAT,
            'C10': BadChannelDescription.FLAT,
            'C17': BadChannelDescription.FLAT,
            'C18': BadChannelDescription.FLAT,
            'C25': BadChannelDescription.FLAT,
            'C26': BadChannelDescription.FLAT,
            'C33': BadChannelDescription.FLAT,
            'C34': BadChannelDescription.FLAT,
            'C41': BadChannelDescription.FLAT,
            'C42': BadChannelDescription.FLAT
        }

        # add bad resected channels
        if session in ['postsurgery']:
            bad_chs = []
            bad_chs.extend([f'C{idx}' for idx in range(1, 6)] + \
                        [f'C{idx}' for idx in range(9, 14)] + \
                        [f'C{idx}' for idx in range(17, 20)] + \
                        [f'C{idx}' for idx in range(25, 27)] + \
                        [f'C{idx}' for idx in range(33, 37)] + \
                        ['C20', 'C25', 'C26', 'C27', 'C28', 'C41', 'C42', 'C47'])
            bad_chs.extend(depths)
            for ch in bad_chs:
                bads[ch] = BadChannelDescription.FLAT

    elif subject == 'E5':
        grids = [f'C{idx}' for idx in range(1, 49)]
        depths = [f'ML{idx}' for idx in range(1, 7)] + \
                 [f'F3C{idx}' for idx in range(1, 7)] + \
                 [f'F1OF{idx}' for idx in range(1, 7)]
        resected = ['C19', 'C20', 'C21',
                    'C27', 'C28', 'C29', 'C36', 'C37']
        bads = ['C2', 'C7', 'C16'] + [f'C{idx}' for idx in range(10, 15)] + \
               [f'C{idx}' for idx in range(18, 22)] + \
               [f'C{idx}' for idx in range(26, 30)] + \
               [f'C{idx}' for idx in range(34, 38)]

    elif subject == 'E6':
        grids = [f'C{idx}' for idx in range(1, 49)] + [f'C{idx}' for idx in range(50, 57)]
        depths = [f'1D{idx}' for idx in range(1, 7)] + \
                 [f'2D{idx}' for idx in range(1, 7)] + \
                 [f'3D{idx}' for idx in range(1, 7)]
        resected = ['C19', 'C20', 'C28', 'C29', 'C37', 'C38']
        bads = ['C15', 'C49'] + [f'C{idx}' for idx in range(17, 21)] + \
               [f'C{idx}' for idx in range(24, 30)] + \
               [f'C{idx}' for idx in range(33, 39)] + \
               [f'C{idx}' for idx in range(52, 55)]

    elif subject == 'E7':
        grids = [f'C{idx}' for idx in range(1, 65)]
        depths = [f'AD{idx}' for idx in range(1, 7)] + \
                 [f'HD{idx}' for idx in range(1, 7)] + \
                 [f'TOD{idx}' for idx in range(1, 7)] + \
                 [f'POD{idx}' for idx in range(1, 7)] + \
                 [f'FOD{idx}' for idx in range(1, 7)] + \
                 [f'ATPS{idx}' for idx in range(1, 7)] + \
                 [f'ABTS{idx}' for idx in range(1, 7)] + \
                 [f'PBTS{idx}' for idx in range(1, 7)]

        resected = ['C41', 'C42', 'C49', 'C50', 'C51',
                    'C57', 'C58', 'C59', 'C60']
        bads = ['C12', 'C24', 'C38', 'C41', 'C42', 'C49', 'C50', 'C51'] + \
               [f'C{idx}' for idx in range(57, 61)]

    # from pprint import pprint
    # ch_names = [ch for ch in list(raw.info['ch_names'])]
    # print('This shold work!')
    # print('F10F1' in ch_names)
    # # print(ch_names)
    # for ch in depths:
    #     if str(ch) not in ch_names:
    #         print('WTF??')
    #         print(ch)
    #         import numpy as np
    #         print(np.where(np.array(raw.info['ch_names']).astype(str) == 'F10F1'))
    # print(raw.info)
    # pprint(raw.info['ch_names'][50:])
    raw.set_channel_types({ch: 'ecog' for ch in grids})
    raw.set_channel_types({ch: 'seeg' for ch in depths})
    # not_set = []
    # for ch in grids:
    #     try:
    #         raw.set_channel_types({ch: 'ecog'})
    #     except ValueError as e:
    #         not_set.append(ch)
    # for ch in depths:
    #     try:
    #         raw.set_channel_types({ch: 'seeg'})
    #     except ValueError as e:
    #         not_set.append(ch)
    # print(f'did not set these channesl for {raw}: ', not_set)
    # raw.set_channel_types({ch: 'seeg' for ch in depths})
    raw.info['bads'].extend(bads.keys())
    raw.info['bads_description'] = bads

    return resected


def write_edf_to_bids(edf_fpath: [str, Path],
                      bids_kwargs: Dict, bids_root: [str, Path],
                      line_freq: int = 60) -> Dict:
    """Write EDF (.edf) files to BIDS format.

    This will convert files to BrainVision (.vhdr, .vmrk, .eeg) data format
    in the BIDS-compliant file-layout.

    The passed in BIDS keyword arguments must adhere to a certain standard.
    Namely, they should include::

        - ``subject``
        - ``session``
        - ``task``
        - ``acquisition``
        - ``run``
        - ``datatype``

    Parameters
    ----------
    edf_fpath : str | Path
    bids_kwargs : dict
    bids_root : str | Path
    line_freq : int

    Returns
    -------
    status_dict : dict
        The resulting status of the BIDS conversion.
    """
    # do check on BIDS kwarg parameters
    entities = _check_bids_parameters(bids_kwargs)

    # load in EDF file
    raw = read_raw_edf(edf_fpath)

    # set channel types
    resected_chs = _set_ch_types(raw, entities.get('subject'),
                                 entities.get('session'))

    # channel text scrub
    raw = _channel_text_scrub(raw)

    # look for bad channels
    bad_chs = _look_for_bad_channels(raw.ch_names)
    raw.info['bads'].extend(bad_chs)

    # set line freq
    raw.info['line_freq'] = line_freq

    # construct the BIDS path that the file will get written to
    bids_path = BIDSPath(**entities, root=bids_root)

    # Get acceptable range of days back and pick random one
    daysback_min, daysback_max = get_anonymization_daysback(raw)
    daysback = random.randrange(daysback_min, daysback_max)
    anonymize = dict(daysback=0, keep_his=False)

    # write to BIDS based on path
    output_bids_path = write_raw_bids(raw, bids_path=bids_path,
                                      anonymize=anonymize,
                                      overwrite=True, verbose=False)

    # add resected channels to description
    channels_tsv_fname = output_bids_path.copy().update(suffix='channels',
                                                        extension='.tsv')

    # update which channels were resected
    for ch in resected_chs:
        _update_sidecar_tsv_byname(channels_tsv_fname, ch,
                                   'description', 'resected')

    # TODO: update which channels were SOZ

    # update status description
    for ch, description in raw.info['bads_description'].items():
        _update_sidecar_tsv_byname(channels_tsv_fname, ch,
                                   'status_description', description)

    # output status dictionary
    status_dict = {
        'status': 1,
        'output_fname': output_bids_path.basename,
        'original_fname': os.path.basename(edf_fpath)
    }
    return status_dict


if __name__ == "__main__":
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        bids_root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")
        source_dir = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/sourcedata")

    elif WORKSTATION == "lab":
        bids_root = Path("/home/adam2392/hdd2/epilepsy_bids/")
        source_dir = Path("/home/adam2392/hdd2/epilepsy_bids/sourcedata")

    # define BIDS identifiers
    modality = "ecog"
    task = "ictal"
    session = "presurgery"
    datatype = 'ieeg'

    # the root of the BIDS dataset
    root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")

    # define BIDS entities
    SUBJECTS = [
        'pt1', 'pt2', 'pt3',  # NIH
        'jh103', 'jh105',  # JHH
        # 'umf001', 'umf002', 'umf003', 'umf004', 'umf005',  # UMF
        # 'la00', 'la01', 'la02', 'la03', 'la04', 'la05', 'la06',
        # 'la07'
    ]

    session = "presurgery"  # only one session

    # pre, Sz, Extraoperative, post
    task = "interictal"
    acquisition = "ecog"
    datatype = "ieeg"
    extension = ".vhdr"

    # regex pattern for the files is:
    for subject in SUBJECTS:
        search_str = f'{subject}*.edf'
        filepaths = source_folder.rglob(search_str)

        # get a list of filepaths for each "task"
        task_filepaths = collections.defaultdict(list)
        for idx, fpath in enumerate(filepaths):
            subject = fpath.name.split('_')[0]
            if subject not in subject_ids:
                print('wtf?')
                continue
            task = fpath.name.split('_')[1].split('.')[0]
            task_filepaths[task].append(fpath)

        for task, filepaths in task_filepaths.items():
            for run_id, fpath in enumerate(filepaths):
                subject = fpath.name.split('_')[0]
                if subject not in subject_ids:
                    print('wtf?')
                    continue

                # get the next available run
                run_id = run_id + 1

                bids_kwargs = {
                    'subject': subject,
                    'session': session,
                    'task': task,
                    'acquisition': modality,
                    'run': run_id,
                    'datatype': datatype,
                    'suffix': datatype,
                }
                print(bids_kwargs
                      )
                # run main bids conversion
                output_dict = write_edf_to_bids(edf_fpath=fpath,
                                                bids_kwargs=bids_kwargs,
                                                bids_root=bids_root, line_freq=60)
                bids_fname = output_dict['output_fname']

                # append scans original filenames
                append_original_fname_to_scans(
                    fpath.name, bids_root, bids_fname
                )

                bids_path = BIDSPath(**bids_kwargs, root=bids_root)
                print(bids_path.fpath)
                raw = read_raw_bids(bids_path)
