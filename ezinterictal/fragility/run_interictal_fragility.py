from logging import warn
from pathlib import Path
from typing import overload
import warnings
from eztrack.utils.config import FiguresConfig
import mne
from eztrack import lds_raw_fragility
from eztrack.fragility.fragility import (
    state_perturbation_derivative,
    state_lds_derivative,
)
from eztrack.io import deriv_info, read_derivative_npy, match_derivative
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.io.read_datasheet import read_clinical_excel
from eztrack.preprocess import preprocess_ieeg
from eztrack.utils import ClinicalContactColumns, set_log_level
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals

from ezinterictal.io.read import load_data

warnings.filterwarnings("ignore")
set_log_level("ERROR")


def load_raw_ieeg_data(bids_path, resample_sfreq=None, verbose=True):
    # load in the data
    raw = read_raw_bids(bids_path, verbose=False)
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False)
    raw = raw.drop_channels(raw.info["bads"])
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
    overwrite=False,
):
    # load in the raw data
    model_params = {
        "winsize": 500,
        "stepsize": 250,
        "radius": 1.5,
        "method_to_use": "pinv",
        "perturb_type": "C",
        "normalize": False,
        "l2penalty": 1e-7,
    }

    # run heatmap
    perturb_deriv, state_arr_deriv, delta_vecs_arr_deriv = lds_raw_fragility(
        raw, reference=reference, return_all=True, **model_params
    )

    print("Saving files to: ")
    print(colperturb_deriv_fpath)
    print(state_deriv_fpath)
    print(coldeltavecs_deriv_fpath)
    perturb_deriv.save(
        deriv_path / colperturb_deriv_fpath.basename, overwrite=overwrite
    )
    state_arr_deriv.save(deriv_path / state_deriv_fpath.basename, overwrite=overwrite)
    delta_vecs_arr_deriv.save(
        deriv_path / coldeltavecs_deriv_fpath.basename, overwrite=overwrite
    )

    return state_arr_deriv, perturb_deriv, delta_vecs_arr_deriv


def plot_heatmap(
    subject,
    perturb_deriv,
    perturb_deriv_fpath,
    excel_fpath,
    figures_path,
):
    # normalize and plot heatmap
    if excel_fpath is not None:
        # read in the dataframe of clinical datasheet
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        # extract the SOZ channels
        soz_chs = pat_dict[ClinicalContactColumns.SOZ_CONTACTS.value]
        epz_chs = pat_dict[ClinicalContactColumns.SPREAD_CONTACTS.value]
        rz_chs = pat_dict[ClinicalContactColumns.RESECTED_CONTACTS.value]

    perturb_deriv.normalize()
    fig_basename = Path(perturb_deriv_fpath.basename).with_suffix(".pdf")
    print(f"Saving figure to {figures_path / fig_basename}.")
    perturb_deriv.plot_heatmap(
        cbarlabel="Fragility",
        cmap="turbo",
        soz_chs=soz_chs,
        figure_fpath=(figures_path / fig_basename),
    )


def _check_channel_count(bids_path, deriv_fpath):
    raw = read_raw_bids(bids_path)
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False)
    raw = raw.drop_channels(raw.info["bads"])

    print(deriv_fpath)
    deriv = read_derivative_npy(deriv_fpath)
    if not all(ch in raw.ch_names for ch in deriv.ch_names):
        return bids_path.subject
    return None


def main():
    mne.set_log_level("ERROR")
    # bids root to write BIDS data to
    # the root of the BIDS dataset
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_interictal/")
    root = Path("/home/adam2392/hdd/Dropbox/epilepsy_interictal")
    output_dir = root / "derivatives"

    figures_dir = output_dir / "figures"

    # path to excel layout file - would be changed to the datasheet locally
    excel_fpath = root / "sourcedata/ieeg_database_all.xlsx"

    task = "interictal"
    session = "extraoperative"  # only one session
    datatype = "ieeg"
    suffix = "ieeg"
    extension = ".edf"
    reference = ""
    overwrite = False

    l2_penalty = 1e-7

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")

    # get list of problematic subs
    problematic_subs = []

    for subject in all_subjects:
        if any(
            char in subject
            for char in [
                # 'jh',
                # 'pt', 'umf',
                # 'upmc',
                # 'NIH',
                # 'PY',
                # 'nl', 'la', 'tvb',
                # 'rns',
                #  'kumc',
                #  'upmc'
            ]
        ):
            continue

        deriv_chain = (
            Path("originalsampling")
            / "fragility"
            / "win500"
            / reference
            / f"l2-{l2_penalty}"
            / f"sub-{subject}"
        )
        deriv_path = output_dir / deriv_chain
        heatmap_path = (
            figures_dir
            / "originalsampling"
            / "win500"
            / f"l2-{l2_penalty}"
            / "fragility"
            / reference
        )

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
        print(f"Found {runs} runs for {task} task for {subject}.")

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

            # run fragility analysis
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

            # check issue
            # _subject = _check_channel_count(bids_path, deriv_path / colperturb_deriv_fpath.basename)
            # problematic_subs.append(_subject)
            # continue

            # search for col perturb derivative
            deriv_basename = coldeltavecs_deriv_fpath.basename
            print(f"Searching for {deriv_basename}")
            fpaths = list(deriv_path.glob(deriv_basename))
            if not overwrite and len(fpaths) > 0:
                print(
                    f"Found derivative for {deriv_basename} already inside {deriv_path}"
                )
                continue

            # load in the raw data
            raw = load_raw_ieeg_data(bids_path)

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
                overwrite=overwrite,
            )

            # save heatmaps
            heatmap_path.mkdir(exist_ok=True, parents=True)
            plot_heatmap(
                subject,
                perturb_deriv,
                colperturb_deriv_fpath,
                excel_fpath,
                heatmap_path,
            )
    print("\n\nThese subjects need to be re-run:")
    print(problematic_subs)


def main_viz():
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
    reference = "av"
    overwrite = False

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")

    # get list of problematic subs
    problematic_subs = []

    for subject in all_subjects:
        # if any(char in subject for char in [
        #     'jh','pt', 'umf',
        #     # 'upmc',
        #     'NIH', 'PY',
        #     # 'nl', 'la', 'tvb',
        #     # # 'rns',
        #      'kumc',
        #     #  'upmc'
        #  ]):
        #     continue
        deriv_chain = (
            Path("originalsampling")
            / "win500"
            / "fragility"
            / reference
            / f"sub-{subject}"
        )
        deriv_path = output_dir / deriv_chain
        heatmap_path = (
            figures_dir / "originalsampling" / "win500" / "fragility" / reference
        )

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

            # run fragility analysis
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
            # if not overwrite and len(fpaths) > 0:
            #     print(
            #         f"Found derivative for {deriv_basename} already inside {deriv_path}"
            #     )
            #     continue

            perturb_deriv = read_derivative_npy(
                deriv_path / colperturb_deriv_fpath.basename, source_check=False
            )
            perturb_deriv.load_data()

            # save heatmaps
            print(f"Plotting heatmap at {heatmap_path}")
            heatmap_path.mkdir(exist_ok=True, parents=True)
            plot_heatmap(
                subject,
                perturb_deriv,
                colperturb_deriv_fpath,
                excel_fpath,
                heatmap_path,
            )
    print("\n\nThese subjects need to be re-run:")
    print(problematic_subs)


if __name__ == "__main__":
    main()
    # main_viz()
