from pathlib import Path
from typing import Union
import json

from mne_bids.path import BIDSPath, get_entities_from_fname, _find_matching_sidecar
from mne_bids.tsv_handler import _from_tsv, _to_tsv

from ezinterictal.bids.utils import _replace_ext


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
        bids_path, suffix="scans.tsv", on_error="ignore"
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
