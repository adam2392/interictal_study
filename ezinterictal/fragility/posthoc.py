import os
from pathlib import Path

import numpy as np
from eztrack import read_result_eztrack
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.io.derivative import read_derivative
from eztrack.io.read_datasheet import read_clinical_excel
from eztrack.utils import ClinicalContactColumns
from mne_bids.path import _parse_ext, get_entity_vals


def read_perturbation_result(deriv_path, source_basename, description):
    if description not in [
        DERIVATIVETYPES.ROWPERTURB_MATRIX.value,
        DERIVATIVETYPES.PERTURB_MATRIX.value,
        DERIVATIVETYPES.STATE_MATRIX.value,
    ]:
        raise RuntimeError(
            f"Perturbation matrix derivative description is only "
            f"one of accepted values."
        )

    source_basename, ext = _parse_ext(source_basename)
    source_basename = source_basename + ".json"
    deriv_basename = _add_desc_to_bids_fname(
        source_basename, description=description, verbose=False
    )
    deriv_fpath = deriv_path / deriv_basename

    result = read_result_eztrack(
        deriv_fname=deriv_fpath, description=description, normalize=False
    )

    return result


def run_svd_viz(heatmap, ch_names, fname=None):
    # create a visualization of the heatmap with just svd
    U, S, VT = np.linalg.svd(heatmap, full_matrices=False)

    # truncate singular values
    S[2:] = 0

    # recombine data
    heatmap_svd = np.dot(U, np.dot(np.diag(S), VT))

    # plot it
    # fig, ax = plt.subplots()
    # ax.plot
    return heatmap_svd


def run_scatterplot_viz(col_pert_deriv, row_pert_deriv, fname=None):
    pass


def run_rf_exp():
    # average the heatmap, or summarize the statistics
    pass


def run_logistic_exp():
    pass


def run_viz_heatmap(
    deriv_path, subject, figures_path, desc, excel_fpath=None, ignore_runs=None
):
    if excel_fpath is not None:
        # read in the dataframe of clinical datasheet
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        # extract the SOZ channels
        soz_chs = pat_dict[ClinicalContactColumns.SOZ_CONTACTS.value]
        epz_chs = pat_dict[ClinicalContactColumns.SPREAD_CONTACTS.value]
        rz_chs = pat_dict[ClinicalContactColumns.RESECTED_CONTACTS.value]

    deriv_path = deriv_path / f"sub-{subject}"
    figures_path = figures_path / f"sub-{subject}"

    # get all file paths
    fpaths = deriv_path.glob(f"*desc-{desc}*.npy")
    if ignore_runs is not None:
        fpaths = [
            fpath
            for fpath in fpaths
            if not f"run-{run}" in str(fpath)
            for run in ignore_runs
        ]

    # print(f'Going to plot these file paths: {list(fpaths)}')

    for fname in fpaths:
        figure_fpath = figures_path / Path(fname).with_suffix(".pdf").name
        plot_kwargs = {
            "cmap": "turbo",
            "soz_chs": soz_chs,
            "cbarlabel": "Fragility",
            "figure_fpath": figure_fpath,
        }

        # load in the data
        derivative = read_derivative(fname)
        derivative.normalize()
        print(f"Plotting heatmap at {figure_fpath}")

        derivative.plot_heatmap(**plot_kwargs)


if __name__ == "__main__":
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        # the root of the BIDS dataset
        root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
        output_dir = root / "derivatives" / "interictal"

        figures_dir = output_dir / "figures"

        # path to excel layout file - would be changed to the datasheet locally
        excel_fpath = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )
    elif WORKSTATION == "lab":
        root = Path("/home/adam2392/hdd/epilepsy_bids/")
        excel_fpath = Path(
            "/home/adam2392/hdd/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )

        # output directory
        output_dir = (
            Path("/home/adam2392/hdd/epilepsy_bids") / "derivatives" / "interictal"
        )

        # figures directory
        figures_dir = output_dir / "figures"

    reference = "monopolar"
    sfreq = 1000  # either resample or don't
    description = "perturbmatrix"

    deriv_path = output_dir / f"{sfreq}Hz" / "fragility" / reference
    figures_path = (
        figures_dir
        # / 'nodepth'
        / f"{sfreq}Hz"
        / "fragility"
        / reference
    )

    # get the runs for this subject
    all_subjects = get_entity_vals(deriv_path, "subject")
    for subject in all_subjects:
        print(f"Plotting for {subject}")
        run_viz_heatmap(
            deriv_path=deriv_path,
            subject=subject,
            desc=description,
            figures_path=figures_path,
            excel_fpath=excel_fpath,
        )
