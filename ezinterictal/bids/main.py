from logging import warn
from typing import Dict
from pathlib import Path
import random
import sys
import warnings
from natsort import natsorted

warnings.filterwarnings("ignore")

from mne.io import read_raw_edf
from mne_bids.path import BIDSPath, get_entity_vals
from mne_bids import get_anonymization_daysback, write_raw_bids

from eztrack.preprocess.excel import (
    add_subject_metadata_from_excel,
    annotate_chs_from_excel,
)

from ezinterictal.bids.write import append_original_fname_to_scans
from ezinterictal.bids.utils import (
    _channel_text_scrub,
    _look_for_bad_channels,
)


def write_edf_to_bids(edf_fpath, bids_path, line_freq: int = 60, ch_type=None) -> Dict:
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
    # load in EDF file
    raw = read_raw_edf(edf_fpath)

    # set channel types
    datatype = bids_path.datatype
    if ch_type is None:
        ch_type = "seeg"
    raw.set_channel_types({ch: ch_type for ch in raw.ch_names})

    # channel text scrub
    raw = _channel_text_scrub(raw)

    # look for bad channels
    bad_chs = _look_for_bad_channels(raw.ch_names, datatype=datatype)
    raw.info["bads"].extend(bad_chs)

    # set line freq
    raw.info["line_freq"] = line_freq

    # Get acceptable range of days back and pick random one
    daysback_min, daysback_max = get_anonymization_daysback(raw)
    daysback = random.randrange(daysback_min, daysback_max)
    anonymize = dict(daysback=daysback, keep_his=False)

    # write to BIDS based on path
    output_bids_path = write_raw_bids(
        raw,
        bids_path=bids_path,
        anonymize=anonymize,
        overwrite=True,
        verbose=False,
        # format='EDF'
    )

    # output status dictionary
    return output_bids_path


def main():
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_interictal/")
    source_dir = root / "sourcedata"
    excel_fpath = source_dir / "ieeg_database_all.xlsx"
    source_dir = Path("/Users/adam2392/OneDrive - Johns Hopkins/neuropace/sourcedata/")

    task = "interictal"
    session = "extraoperative"
    datatype = "ieeg"
    suffix = "ieeg"
    extension = ".edf"

    ch_type = "seeg"

    # do 'kumc' separately
    fpaths = []
    for site in [
        "umfrns"
        # 'jhu',
        # 'nih',
        # 'cclinic',
        # 'umf',
        # 'upmc',
        # 'kumc'
    ]:
        site_dir = source_dir / site

        # get all edf files
        _fpaths = natsorted(list(site_dir.rglob("*interictal*.edf")))
        fpaths.extend(_fpaths)

        if site == "jhu":
            splitter = "_"
        elif site == "nih":
            splitter = "-"
        elif site == "upmc":
            splitter = "."
        elif site == "cclinic":
            splitter = "_"
        elif site == "umfrns":
            splitter = "_"

        # subjects = [f.split(splitter)[0] for f in _fpaths]

        print(f"Looking at {site}")
        # now analyze that file
        for fpath in _fpaths:
            if site == "kumc":
                old_subject = fpath.parent.name.split("-")[1]
                subject = "kumc" + old_subject.split("pt")[1]
            elif site == "umf":
                subject = fpath.parent.name
            elif site == "umfrns":
                subject = "rns" + fpath.name.split(splitter)[1]
            else:
                subject = fpath.name.split(splitter)[0]

            if "aw" in fpath.name.lower():
                task = "interictalawake"
            elif "aslp" in fpath.name.lower():
                task = "interictalasleep"
            else:
                task = "interictal"

            # if subject == 'umf004':
            #     ch_type = 'seeg'
            # else:
            #     ch_type = 'ecog'

            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                run="01",
                root=root,
                suffix=suffix,
                datatype=datatype,
                extension=extension,
            )
            print(f"\n\nConverting {fpath} to {bids_path}...\n\n")
            if len(bids_path.match()) > 0:
                print("\n\nSkipping this file... ")
                bids_path.run = "02"
                continue

            # run main bids conversion
            bids_fname = write_edf_to_bids(
                edf_fpath=fpath, bids_path=bids_path, line_freq=60, ch_type=ch_type
            )

            # append scans original filenames
            append_original_fname_to_scans(fpath.name, root, bids_fname.basename)

            # add subject specific metadata
            add_subject_metadata_from_excel(root, subject, excel_fpath=excel_fpath)
            # annotate_chs_from_excel(root, subject, excel_fpath=excel_fpath)


def _get_nih_files():
    pass


def main_fix_chs():
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_interictal/")
    source_dir = root / "sourcedata"
    excel_fpath = source_dir / "ieeg_database_all.xlsx"

    subjects = get_entity_vals(root, "subject")
    for subject in subjects:
        # if 'rns' not in subject:
        #     continue
        annotate_chs_from_excel(
            root, subject, excel_fpath=excel_fpath, extension=".edf"
        )


if __name__ == "__main__":
    # main()
    main_fix_chs()
