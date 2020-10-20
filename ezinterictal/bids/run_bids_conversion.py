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
import numpy as np
from mne.io import read_raw_edf
from mne.utils import warn
from mne_bids import write_raw_bids, read_raw_bids, get_anonymization_daysback, mark_bad_channels
from mne_bids.path import (BIDSPath, get_entities_from_fname,
                           _find_matching_sidecar, _parse_ext,
                           get_entity_vals)
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from ezinterictal.bids.utils import (_replace_ext, BadChannelDescription,
                                     _channel_text_scrub, _check_bids_parameters,
                                     _look_for_bad_channels, _update_sidecar_tsv_byname)
from eztrack.preprocess.excel import (add_subject_metadata_from_excel, annotate_chs_from_excel)


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
        bids_path, suffix="scans.tsv", on_error='ignore'
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


# def _set_ch_types(raw, subject, session):
#     raw.set_channel_types({ch: 'misc' for ch in raw.ch_names})
#
#     if subject == 'E1':
#         grids = [f'C{idx}' for idx in range(1, 65)]
#         depths = [f'SAD{idx}' for idx in range(1, 7)] + \
#                  [f'SPD{idx}' for idx in range(1, 7)] + \
#                  [f'MPOD{idx}' for idx in range(1, 7)] + \
#                  [f'IAD{idx}' for idx in range(1, 7)] + \
#                  [f'MAOD{idx}' for idx in range(1, 7)] + \
#                  [f'IPD{idx}' for idx in range(1, 7)]
#         resected = ['C19', 'C20', 'C27', 'C28']
#         bads = ['C15'] + [f'C{idx}' for idx in range(17, 21)] + \
#                [f'C{idx}' for idx in range(24, 29)] + \
#                [f'C{idx}' for idx in range(34, 37)] + \
#                ['C51', 'C57', 'C59']
#         bads = {
#             'C15': BadChannelDescription.HIGHFREQ,
#             # 'C55': BadChannelDescription.FLAT,
#             # 'C56': BadChannelDescription.FLAT,
#             'C57': BadChannelDescription.HIGHFREQ,
#             # 'C63': BadChannelDescription.FLAT,
#             # 'C64': BadChannelDescription.FLAT,
#         }
#
#         # add bad resected channels
#         if session in ['postsurgery', 'intraoperative']:
#             bad_chs = []
#             bad_chs.extend(['C15', 'C17', 'C18', 'C19', 'C20'] + \
#                            [f'C{idx}' for idx in range(25, 29)] + \
#                            ['C34', 'C51', 'C57', 'C59'])
#             bad_chs.extend(depths)
#             for ch in bad_chs:
#                 bads[ch] = BadChannelDescription.FLAT
#
#     elif subject == 'E2':
#         grids = [f'C{idx}' for idx in range(1, 65)]
#         depths = []
#         resected = ['C1', 'C2']
#         bads = [f'C{idx}' for idx in range(1, 11)]
#         bads = []
#     elif subject == 'E3':
#         grids = [f'C{idx}' for idx in range(1, 65)]
#         depths = \
#             [f'ASD{idx}' for idx in range(1, 7)] + \
#             [f'PD{idx}' for idx in
#              range(1, 7)]  # + [f'PS{idx}' for idx in range(1, 5)] + # [f'AID{idx}' for idx in range(1, 7)] + \
#         resected = ['C3', 'C4', 'C11', 'C12', 'C19',
#                     'C20', 'C27', 'C28']
#         bads = [f'C{idx}' for idx in range(1, 6)] + \
#                [f'C{idx}' for idx in range(9, 15)] + \
#                [f'C{idx}' for idx in range(17, 24)] + \
#                [f'C{idx}' for idx in range(25, 32)] + ['C51']
#
#         bads = {
#             'C51': BadChannelDescription.FLAT,
#         }
#     elif subject == 'E4':
#         grids = [f'C{idx}' for idx in range(1, 49)]
#         depths = [f'F2AL{idx}' for idx in range(1, 7)] + \
#                  [f'F2BC{idx}' for idx in range(1, 7)] + \
#                  [f'F2CL{idx}' for idx in range(1, 7)]
#         resected = ['C19', 'C20', 'C27', 'C28', 'C35', 'C36']
#         bads = [f'C{idx}' for idx in range(3, 6)] + \
#                [f'C{idx}' for idx in range(11, 14)] + \
#                ['C19', 'C20', 'C27', 'C28', 'C35', 'C36',
#                 'C47']
#
#         bads = {
#             'C1': BadChannelDescription.FLAT,
#             'C2': BadChannelDescription.FLAT,
#             'C3': BadChannelDescription.HIGHFREQ,
#             'C9': BadChannelDescription.FLAT,
#             'C10': BadChannelDescription.FLAT,
#             'C17': BadChannelDescription.FLAT,
#             'C18': BadChannelDescription.FLAT,
#             'C25': BadChannelDescription.FLAT,
#             'C26': BadChannelDescription.FLAT,
#             'C33': BadChannelDescription.FLAT,
#             'C34': BadChannelDescription.FLAT,
#             'C41': BadChannelDescription.FLAT,
#             'C42': BadChannelDescription.FLAT
#         }
#
#         # add bad resected channels
#         if session in ['postsurgery']:
#             bad_chs = []
#             bad_chs.extend([f'C{idx}' for idx in range(1, 6)] + \
#                         [f'C{idx}' for idx in range(9, 14)] + \
#                         [f'C{idx}' for idx in range(17, 20)] + \
#                         [f'C{idx}' for idx in range(25, 27)] + \
#                         [f'C{idx}' for idx in range(33, 37)] + \
#                         ['C20', 'C25', 'C26', 'C27', 'C28', 'C41', 'C42', 'C47'])
#             bad_chs.extend(depths)
#             for ch in bad_chs:
#                 bads[ch] = BadChannelDescription.FLAT
#
#     elif subject == 'E5':
#         grids = [f'C{idx}' for idx in range(1, 49)]
#         depths = [f'ML{idx}' for idx in range(1, 7)] + \
#                  [f'F3C{idx}' for idx in range(1, 7)] + \
#                  [f'F1OF{idx}' for idx in range(1, 7)]
#         resected = ['C19', 'C20', 'C21',
#                     'C27', 'C28', 'C29', 'C36', 'C37']
#         bads = ['C2', 'C7', 'C16'] + [f'C{idx}' for idx in range(10, 15)] + \
#                [f'C{idx}' for idx in range(18, 22)] + \
#                [f'C{idx}' for idx in range(26, 30)] + \
#                [f'C{idx}' for idx in range(34, 38)]
#
#     elif subject == 'E6':
#         grids = [f'C{idx}' for idx in range(1, 49)] + [f'C{idx}' for idx in range(50, 57)]
#         depths = [f'1D{idx}' for idx in range(1, 7)] + \
#                  [f'2D{idx}' for idx in range(1, 7)] + \
#                  [f'3D{idx}' for idx in range(1, 7)]
#         resected = ['C19', 'C20', 'C28', 'C29', 'C37', 'C38']
#         bads = ['C15', 'C49'] + [f'C{idx}' for idx in range(17, 21)] + \
#                [f'C{idx}' for idx in range(24, 30)] + \
#                [f'C{idx}' for idx in range(33, 39)] + \
#                [f'C{idx}' for idx in range(52, 55)]
#
#     elif subject == 'E7':
#         grids = [f'C{idx}' for idx in range(1, 65)]
#         depths = [f'AD{idx}' for idx in range(1, 7)] + \
#                  [f'HD{idx}' for idx in range(1, 7)] + \
#                  [f'TOD{idx}' for idx in range(1, 7)] + \
#                  [f'POD{idx}' for idx in range(1, 7)] + \
#                  [f'FOD{idx}' for idx in range(1, 7)] + \
#                  [f'ATPS{idx}' for idx in range(1, 7)] + \
#                  [f'ABTS{idx}' for idx in range(1, 7)] + \
#                  [f'PBTS{idx}' for idx in range(1, 7)]
#
#         resected = ['C41', 'C42', 'C49', 'C50', 'C51',
#                     'C57', 'C58', 'C59', 'C60']
#         bads = ['C12', 'C24', 'C38', 'C41', 'C42', 'C49', 'C50', 'C51'] + \
#                [f'C{idx}' for idx in range(57, 61)]
#
#     # from pprint import pprint
#     # ch_names = [ch for ch in list(raw.info['ch_names'])]
#     # print('This shold work!')
#     # print('F10F1' in ch_names)
#     # # print(ch_names)
#     # for ch in depths:
#     #     if str(ch) not in ch_names:
#     #         print('WTF??')
#     #         print(ch)
#     #         import numpy as np
#     #         print(np.where(np.array(raw.info['ch_names']).astype(str) == 'F10F1'))
#     # print(raw.info)
#     # pprint(raw.info['ch_names'][50:])
#     raw.set_channel_types({ch: 'ecog' for ch in grids})
#     raw.set_channel_types({ch: 'seeg' for ch in depths})
#     # not_set = []
#     # for ch in grids:
#     #     try:
#     #         raw.set_channel_types({ch: 'ecog'})
#     #     except ValueError as e:
#     #         not_set.append(ch)
#     # for ch in depths:
#     #     try:
#     #         raw.set_channel_types({ch: 'seeg'})
#     #     except ValueError as e:
#     #         not_set.append(ch)
#     # print(f'did not set these channesl for {raw}: ', not_set)
#     # raw.set_channel_types({ch: 'seeg' for ch in depths})
#     raw.info['bads'].extend(bads.keys())
#     raw.info['bads_description'] = bads
#
#     return resected


def write_edf_to_bids(edf_fpath: [str, Path],
                      bids_kwargs: Dict, bids_root: [str, Path],
                      line_freq: int = 60, datatype: str=None) -> Dict:
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

    nan_indx = np.argwhere(np.isnan(raw.get_data()))
    if nan_indx.any():
        print(nan_indx)
        print(len(nan_indx))

    # set channel types
    # resected_chs = _set_ch_types(raw, entities.get('subject'),
    #                              entities.get('session'))
    if datatype is None:
        datatype = 'ecog'
    raw.set_channel_types({ch: datatype for ch in raw.ch_names})

    # channel text scrub
    raw = _channel_text_scrub(raw)

    # look for bad channels
    bad_chs = _look_for_bad_channels(raw.ch_names, datatype=datatype)
    raw.info['bads'].extend(bad_chs)

    if entities['subject'] == 'pt11' and datatype != 'eeg':
        raw.info['bads'].extend(['RLG12-0', 'RLG12-1', 'RIPS4-0', 'RIPS4-1',])

    # set line freq
    raw.info['line_freq'] = line_freq

    # construct the BIDS path that the file will get written to
    bids_path = BIDSPath(**entities, root=bids_root)

    # Get acceptable range of days back and pick random one
    daysback_min, daysback_max = get_anonymization_daysback(raw)
    daysback = random.randrange(daysback_min, daysback_max)
    anonymize = dict(daysback=0, keep_his=False)

    # write to BIDS based on path
    print(F'HERE ARE THE RAW UNITS: {raw._orig_units}')
    output_bids_path = write_raw_bids(raw, bids_path=bids_path,
                                      anonymize=anonymize,
                                      overwrite=True, verbose=False)

    # add resected channels to description
    # channels_tsv_fname = output_bids_path.copy().update(suffix='channels',
    #                                                     extension='.tsv')

    # update which channels were resected
    # for ch in resected_chs:
    #     _update_sidecar_tsv_byname(channels_tsv_fname, ch,
    #                                'description', 'resected')
    # TODO: update which channels were SOZ

    # update status description
    # for ch, description in raw.info['bads_description'].items():
    #     _update_sidecar_tsv_byname(channels_tsv_fname, ch,
    #                                'status_description', description)

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
        root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_bids/")
        source_dir = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_bids/sourcedata")

        root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
        source_dir = Path("/Users/adam2392/Dropbox/epilepsy_bids/sourcedata")

        # path to excel layout file - would be changed to the datasheet locally
        excel_fpath = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )
    elif WORKSTATION == "lab":
        root = Path("/home/adam2392/hdd2/epilepsy_bids/")
        source_dir = Path("/home/adam2392/hdd2/epilepsy_bids/sourcedata")
        excel_fpath = Path(
            "/home/adam2392/hdd2/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )
    # define BIDS identifiers
    modality = "eeg"
    task = "ictal"
    session = "presurgery"
    datatype = 'eeg'
    extension = ".vhdr"

    # define BIDS entities
    SUBJECTS = [
        'pt1',
        # 'pt2',
        'pt3',
        'pt6',
        'pt7', 'pt8',
        # 'pt9',
        'pt11',
        # 'la09',
        'pt12', 'pt13', 'pt14', 'pt15', 'pt16', 'pt17', # NIH
        # 'jh103', 'jh105',  # JHH
        # 'umf001', 'umf002', 'umf003', 'umf004', 'umf005',  # UMF
        # 'la00', 'la01', 'la02', 'la03', 'la04', 'la05', 'la06',
        # 'la07'
    ]


    # regex pattern for the files is:
    for subject in SUBJECTS:
        source_folder = source_dir / 'nih' / subject / 'scalp'
        print(source_folder)
        search_str = f'*Inter_with_P_3.edf'
        search_str = f'*.edf'
        filepaths = list(source_folder.rglob(search_str))

        subjects = get_entity_vals(root, 'subject')
        ignore_subjects = [sub for sub in subjects if sub != subject]
        runs = get_entity_vals(root, 'run',
                               ignore_subjects=ignore_subjects,
                               ignore_modalities=['ieeg'])
        runs = [int(run) for run in runs]
        runs = []
        print(runs)
        print(f'Found these filepaths {list(filepaths)}')
        for run_id, fpath in enumerate(filepaths):
            # subject = fpath.name.split('_')[0]
            # if subject not in subject_ids:
            #     print('wtf?')
            #     continue

            # get the next available run
            if runs:
                run_id = max(runs) + 1
            else:
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
            print(f'Bids kwargs: {bids_kwargs}')
            # run main bids conversion
            output_dict = write_edf_to_bids(edf_fpath=fpath,
                                            bids_kwargs=bids_kwargs,
                                            bids_root=root, line_freq=60,
                                            datatype=datatype)
            bids_fname = output_dict['output_fname']

            # append scans original filenames
            append_original_fname_to_scans(
                fpath.name, root, bids_fname
            )


    # append metadata to all subjects and their tsv files
    # all_subjects = get_entity_vals(root, 'subject')
    # all_subjects = [
    #     # 'la09',
    #     # 'pt6',
    #     # 'pt7', 'pt8', 'pt10', 'pt11', 'pt12', 'pt13', 'pt14', 'pt15'
    # ]
    # for subject in all_subjects:
    #     # add_subject_metadata_from_excel(root, subject, excel_fpath=excel_fpath)
    #     annotate_chs_from_excel(root, subject, excel_fpath=excel_fpath)