from pathlib import Path
from typing import overload

from eztrack import lds_raw_fragility
from eztrack.fragility.fragility import (
    state_perturbation_derivative,
    state_lds_derivative,
)
from eztrack.io import deriv_info, read_derivative_npy, match_derivative
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.io.read_datasheet import read_clinical_excel
from eztrack.preprocess import preprocess_ieeg
from eztrack.utils import ClinicalContactColumns
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals

from ezinterictal.io.read import load_data


def load_raw_ieeg_data(bids_path, resample_sfreq=None, verbose=True):
    # load in the data
    raw = read_raw_bids(bids_path, verbose=False)
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False)
    raw.load_data()

    if resample_sfreq is not None:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    # pre-process the data using preprocess pipeline
    print("Power Line frequency is : ", raw.info["line_freq"])
    l_freq = 0.5
    h_freq = min(300, raw.info["sfreq"] // 2 - 1)
    raw = preprocess_ieeg(raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose)
    return raw


def run_fragility_analysis(
    raw,
    deriv_path,
    state_deriv_fpath,
    colperturb_deriv_fpath,
    coldeltavecs_deriv_fpath,
    reference="monopolar",
    resample_sfreq=None,
):
    # load in the raw data
    model_params = {
        "winsize": 250,
        "stepsize": 125,
        "radius": 1.5,
        "method_to_use": "pinv",
        "perturb_type": "C",
    }

    # run heatmap
    perturb_deriv, state_arr_deriv, delta_vecs_arr_deriv = lds_raw_fragility(
        raw, reference=reference, return_all=True, **model_params
    )

    print("Saving files to: ")
    print(colperturb_deriv_fpath)
    print(state_deriv_fpath)
    print(coldeltavecs_deriv_fpath)
    perturb_deriv.save(deriv_path / colperturb_deriv_fpath.basename)
    state_arr_deriv.save(deriv_path / state_deriv_fpath.basename)
    delta_vecs_arr_deriv.save(deriv_path / coldeltavecs_deriv_fpath.basename)

    return state_arr_deriv, perturb_deriv, delta_vecs_arr_deriv


def plot_heatmap(
    subject, perturb_deriv, perturb_deriv_fpath, excel_fpath, figures_path
):
    # normalize and plot heatmap
    if plot_heatmap:
        if excel_fpath is not None:
            # read in the dataframe of clinical datasheet
            pat_dict = read_clinical_excel(excel_fpath, subject=subject)
            # extract the SOZ channels
            soz_chs = pat_dict[ClinicalContactColumns.SOZ_CONTACTS.value]
            epz_chs = pat_dict[ClinicalContactColumns.SPREAD_CONTACTS.value]
            rz_chs = pat_dict[ClinicalContactColumns.RESECTED_CONTACTS.value]

        perturb_deriv.normalize()
        fig_basename = perturb_deriv_fpath.fpath.with_suffix(".pdf")
        perturb_deriv.plot_heatmap(
            cbarlabel="Fragility",
            cmap="turbo",
            soz_chs=soz_chs,
            figure_fpath=(figures_path / fig_basename),
        )


def main():
    # bids root to write BIDS data to
    # the root of the BIDS dataset
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_interictal/")
    output_dir = root / "derivatives"

    figures_dir = output_dir / "figures"

    # path to excel layout file - would be changed to the datasheet locally
    excel_fpath = root / "sourcedata/ieeg_database_all.xlsx"

    task = "interictal"
    session = "extraoperative"  # only one session
    datatype = "ieeg"
    suffix = "ieeg"
    extension = ".edf"
    reference = "monopolar"
    overwrite = False

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")

    for subject in all_subjects:
        # if subject not in SUBJECTS:
        #     continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]
        all_tasks = get_entity_vals(
            root,
            "task",
            ignore_subjects=ignore_subs,
        )
        ignore_tasks = [tsk for tsk in all_tasks if tsk != task]

        runs = get_entity_vals(
            root,
            "run",
            ignore_subjects=ignore_subs,
            ignore_tasks=ignore_tasks,
        )
        print(f"Found {runs} runs for {task} task.")

        for idx, run in enumerate(runs):
            # create path for the dataset
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                run=run,
                datatype=datatype,
                root=root,
                extension=extension,
                suffix=suffix,
            )
            print(f"Analyzing {bids_path}")

            # load in the raw data
            raw = load_raw_ieeg_data(bids_path)

            # run fragility analysis
            deriv_chain = (
                Path("originalsampling") / "fragility" / reference / f"sub-{subject}"
            )
            deriv_path = output_dir / deriv_chain
            state_deriv_fpath = bids_path.copy().update(
                check=False,
                suffix=f"desc-{DERIVATIVETYPES.STATE_MATRIX.value}_ieeg",
                extension=".npy",
            )
            colperturb_deriv_fpath = bids_path.copy().update(
                check=False,
                suffix=f"desc-{DERIVATIVETYPES.COLPERTURB_MATRIX.value}_ieeg",
                extension=".npy",
            )
            coldeltavecs_deriv_fpath = bids_path.copy().update(
                check=False,
                suffix=f"desc-{DERIVATIVETYPES.DELTAVECS_MATRIX.value}_ieeg",
                extension=".npy",
            )

            # search for col perturb derivative
            deriv_basename = coldeltavecs_deriv_fpath.basename
            print(f"Searching for {deriv_basename}")
            fpaths = list(deriv_path.glob(deriv_basename))
            if not overwrite and len(fpaths) > 0:
                print(
                    f"Found derivative for {deriv_basename} already inside {deriv_path}"
                )
                continue
            (
                state_arr_deriv,
                perturb_deriv,
                delta_vecs_arr_deriv,
            ) = run_fragility_analysis(
                raw,
                deriv_path=deriv_path,
                state_deriv_fpath=state_deriv_fpath,
                colperturb_deriv_fpath=colperturb_deriv_fpath,
                coldeltavecs_deriv_fpath=coldeltavecs_deriv_fpath,
                reference=reference,
            )

            # save heatmaps
            heatmap_path = (
                figures_dir
                / "originalsampling"
                / "fragility"
                / reference
                / f"sub-{subject}"
            )
            plot_heatmap(
                subject,
                perturb_deriv,
                colperturb_deriv_fpath,
                excel_fpath,
                heatmap_path,
            )


if __name__ == "__main__":
    main()
