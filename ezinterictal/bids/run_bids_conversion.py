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
from mne_bids import (
    write_raw_bids,
    read_raw_bids,
    get_anonymization_daysback,
    mark_bad_channels,
)
from mne_bids.path import (
    BIDSPath,
    get_entities_from_fname,
    _find_matching_sidecar,
    _parse_ext,
    get_entity_vals,
)
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from ezinterictal.bids.utils import (
    _replace_ext,
    BadChannelDescription,
    _channel_text_scrub,
    _check_bids_parameters,
    _look_for_bad_channels,
    _update_sidecar_tsv_byname,
)
from eztrack.preprocess.excel import (
    add_subject_metadata_from_excel,
    annotate_chs_from_excel,
)


def _modify_raw_with_json(
    raw: mne.io.BaseRaw, ch_metadata: Dict, verbose: bool = True
) -> mne.io.BaseRaw:
    # get the bad channels and set them
    bad_chs = ch_metadata["bad_contacts"]

    # set channel types
    ch_types = ch_metadata["ch_types"]
    raw.set_channel_types(ch_types, verbose=verbose)

    # set additional bad channels
    raw.info["bads"].extend(bad_chs)

    return raw


def write_edf_to_bids(
    edf_fpath: [str, Path],
    bids_kwargs: Dict,
    bids_root: [str, Path],
    line_freq: int = 60,
    datatype: str = None,
) -> Dict:
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
    if datatype is None:
        modality = "ecog"
    modality = entities["acquisition"]
    raw.set_channel_types({ch: modality for ch in raw.ch_names})

    # channel text scrub
    raw = _channel_text_scrub(raw)

    # look for bad channels
    bad_chs = _look_for_bad_channels(raw.ch_names, datatype=datatype)
    raw.info["bads"].extend(bad_chs)

    if entities["subject"] == "pt11" and datatype != "eeg":
        raw.info["bads"].extend(
            [
                "RLG12-0",
                "RLG12-1",
                "RIPS4-0",
                "RIPS4-1",
            ]
        )

    # set line freq
    raw.info["line_freq"] = line_freq

    # construct the BIDS path that the file will get written to
    bids_path = BIDSPath(**entities, root=bids_root)

    # Get acceptable range of days back and pick random one
    daysback_min, daysback_max = get_anonymization_daysback(raw)
    daysback = random.randrange(daysback_min, daysback_max)
    anonymize = dict(daysback=0, keep_his=False)

    # write to BIDS based on path
    print(f"HERE ARE THE RAW UNITS: {raw._orig_units}")
    output_bids_path = write_raw_bids(
        raw, bids_path=bids_path, anonymize=anonymize, overwrite=True, verbose=False
    )

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
        "status": 1,
        "output_fname": output_bids_path.basename,
        "original_fname": os.path.basename(edf_fpath),
    }
    return status_dict


if __name__ == "__main__":
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_bids/")
        source_dir = Path(
            "/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_bids/sourcedata"
        )

        # root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
        # source_dir = Path("/Users/adam2392/Dropbox/epilepsy_bids/sourcedata")

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
    modality = "seeg"
    task = "ictal"
    session = "presurgery"
    datatype = "ieeg"
    extension = ".vhdr"

    # define BIDS entities
    SUBJECTS = [
        "la08",
        # 'pt1',
        # # 'pt2',
        # 'pt3',
        # 'pt6',
        # 'pt7', 'pt8',
        # # 'pt9',
        # 'pt11',
        # # 'la09',
        # 'pt12', 'pt13', 'pt14', 'pt15', 'pt16', 'pt17', # NIH
        # 'jh103', 'jh105',  # JHH
        # 'umf001', 'umf002', 'umf003', 'umf004', 'umf005',  # UMF
        # 'la00', 'la01', 'la02', 'la03', 'la04', 'la05', 'la06',
        # 'la07'
    ]

    # regex pattern for the files is:
    for subject in SUBJECTS:
        source_folder = source_dir / "nih" / subject / "scalp"
        source_folder = source_dir / "cleveland" / subject / "seeg"
        print(source_folder)
        # search_str = f'*Inter_with_P_3.edf'
        search_str = f"*.edf"
        filepaths = list(source_folder.rglob(search_str))

        subjects = get_entity_vals(root, "subject")
        ignore_subjects = [sub for sub in subjects if sub != subject]
        runs = get_entity_vals(
            root, "run", ignore_subjects=ignore_subjects, ignore_modalities=["ieeg"]
        )
        runs = [int(run) for run in runs]
        runs = []
        print(runs)
        print(f"Found these filepaths {list(filepaths)}")
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
                "subject": subject,
                "session": session,
                "task": task,
                "acquisition": modality,
                "run": run_id,
                "datatype": datatype,
                "suffix": datatype,
            }
            print(f"Bids kwargs: {bids_kwargs}")
            # run main bids conversion
            output_dict = write_edf_to_bids(
                edf_fpath=fpath,
                bids_kwargs=bids_kwargs,
                bids_root=root,
                line_freq=60,
                datatype=datatype,
            )
            bids_fname = output_dict["output_fname"]

            # append scans original filenames
            append_original_fname_to_scans(fpath.name, root, bids_fname)

    # append metadata to all subjects and their tsv files
    # all_subjects = get_entity_vals(root, 'subject')
    all_subjects = [
        "la08",
        # 'pt6',
        # 'pt7', 'pt8', 'pt10', 'pt11', 'pt12', 'pt13', 'pt14', 'pt15'
    ]
    for subject in all_subjects:
        add_subject_metadata_from_excel(root, subject, excel_fpath=excel_fpath)
        annotate_chs_from_excel(root, subject, excel_fpath=excel_fpath)
