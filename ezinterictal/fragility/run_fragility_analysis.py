from pathlib import Path

from eztrack import lds_raw_fragility
from eztrack.fragility.fragility import (
    state_perturbation_derivative,
    state_lds_derivative,
)
from eztrack.io import read_derivative_npy, match_derivative
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.io.read_datasheet import read_clinical_excel
from eztrack.preprocess import preprocess_ieeg
from eztrack.utils import ClinicalContactColumns
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals

from ezinterictal.io.read import load_data


def load_raw_ieeg_data(
    bids_path, figures_path, resample_sfreq=None, plot_raw=False, verbose=True
):
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

    if plot_raw:
        # plot raw data
        figures_path.mkdir(exist_ok=True, parents=True)
        fig_basename = bids_path.copy().update(extension=".pdf").basename
        scale = 200e-6
        fig = raw.plot(
            decim=10,
            scalings={"ecog": scale, "seeg": scale},
            n_channels=len(raw.ch_names),
        )
        fig.savefig(figures_path / fig_basename)
    return raw


def run_fragility_analysis(
    bids_path,
    reference="monopolar",
    resample_sfreq=None,
    deriv_path=None,
    plot_raw=False,
    plot_heatmap=True,
    figures_path=None,
    excel_fpath=None,
    verbose=True,
    overwrite=False,
):
    subject = bids_path.subject

    # get the root derivative path
    if resample_sfreq is not None:
        sample_chain_name = f"resample{resample_sfreq}"
    else:
        sample_chain_name = "originalsampling"

    deriv_chain = Path(sample_chain_name) / "fragility" / reference / f"sub-{subject}"
    deriv_path = deriv_path / deriv_chain

    # check if data already there
    bids_basename = BIDSPath(
        **bids_path.copy().update(suffix=None, extension=None).entities
    ).basename
    description = DERIVATIVETYPES.COLPERTURB_MATRIX.value
    extension = ".npy"
    search_str = f"*{bids_basename}*desc-{description}_{datatype}{extension}"
    print(search_str)
    fpaths = list(deriv_path.glob(search_str))
    if not overwrite and len(fpaths) > 0:
        print(f"Found derivative for {bids_path} already inside {deriv_path}")
        return

    # load in the raw data
    raw_figures_path = figures_path / deriv_chain
    raw = load_raw_ieeg_data(
        bids_path,
        raw_figures_path,
        resample_sfreq=resample_sfreq,
        plot_raw=plot_raw,
        verbose=verbose,
    )

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

    # save the files
    perturb_deriv_fpath = deriv_path / perturb_deriv.info._expected_basename
    state_deriv_fpath = deriv_path / state_arr_deriv.info._expected_basename
    delta_vecs_deriv_fpath = deriv_path / delta_vecs_arr_deriv.info._expected_basename

    print("Saving files to: ")
    print(perturb_deriv_fpath)
    print(state_deriv_fpath)
    print(delta_vecs_deriv_fpath)
    perturb_deriv.save(perturb_deriv_fpath)
    state_arr_deriv.save(state_deriv_fpath)
    delta_vecs_arr_deriv.save(delta_vecs_deriv_fpath)

    # normalize and plot heatmap
    if plot_heatmap:
        if excel_fpath is not None:
            # read in the dataframe of clinical datasheet
            pat_dict = read_clinical_excel(excel_fpath, subject=subject)
            # extract the SOZ channels
            soz_chs = pat_dict[ClinicalContactColumns.SOZ_CONTACTS.value]
            epz_chs = pat_dict[ClinicalContactColumns.SPREAD_CONTACTS.value]
            rz_chs = pat_dict[ClinicalContactColumns.RESECTED_CONTACTS.value]
        figures_path = (
            figures_path
            / f"{int(raw.info['sfreq'])}Hz"
            / "fragility"
            / reference
            / f"sub-{subject}"
        )

        perturb_deriv.normalize()
        fig_basename = perturb_deriv_fpath.with_suffix(".pdf")
        perturb_deriv.plot_heatmap(
            cbarlabel="Fragility",
            cmap="turbo",
            soz_chs=soz_chs,
            figure_fpath=(figures_path / fig_basename),
        )


def run_perturbation_analysis(
    state_deriv_fpath,
    perturb_type="C",
    plot_heatmap=True,
    figures_path=None,
    excel_fpath=None,
):
    # read in the derivative object
    state_deriv = read_derivative(state_deriv_fpath)

    model_params = {
        "radius": 1.5,
        "perturb_type": perturb_type,
    }

    # run perturbation analysis
    perturb_deriv, deltavecs_deriv = state_perturbation_derivative(
        state_deriv, **model_params
    )

    # save the files
    deriv_path = Path(state_deriv_fpath).parent
    perturb_deriv_fpath = deriv_path / perturb_deriv.info._expected_basename
    delta_vecs_deriv_fpath = deriv_path / deltavecs_deriv.info._expected_basename

    print("Saving files to: ")
    print(perturb_deriv_fpath)
    print(delta_vecs_deriv_fpath)
    perturb_deriv.save(perturb_deriv_fpath, overwrite=True)
    deltavecs_deriv.save(delta_vecs_deriv_fpath, overwrite=False)

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
        fig_basename = perturb_deriv_fpath.with_suffix(".pdf")
        perturb_deriv.plot_heatmap(
            cbarlabel="Fragility",
            cmap="turbo",
            soz_chs=soz_chs,
            figure_fpath=figures_path / fig_basename,
        )


def run_analysis(
    bids_path,
    state_deriv_fname,
    colperturb_deriv_fname,
    # rowperturb_deriv_fname,
    reference="monopolar",
    resample_sfreq=None,
    deriv_root=None,
    figures_path=None,
    excel_fpath=None,
    verbose=True,
    overwrite=False,
):
    subject = bids_path.subject

    # load in raw data
    fig_basename = bids_path.copy().update(extension=".svg", check=False).basename
    figure_fpath = (
        root
        / "derivatives"
        / "figures"
        / "raw"
        / reference
        / f"sub-{subject}"
        / fig_basename
    )
    deriv_chain = Path("fragility") / reference / f"sub-{subject}"
    figures_path = figures_path / deriv_chain
    deriv_path = deriv_root / deriv_chain
    raw = load_data(
        bids_path, resample_sfreq, figure_fpath=figure_fpath, verbose=verbose
    )

    sysid_params = {
        "winsize": 500,
        "stepsize": 250,
        "l2penalty": 0,
        "method_to_use": "pinv",
    }
    perturbation_params = {
        "radius": 1.5,
        "perturb_type": "C",
        # "perturbation_strategy": "univariate",
    }

    # if state derivative exists, and not overwrite skip
    state_deriv_fpath = match_derivative(
        deriv_path, state_deriv_fname, ".json", verbose=verbose
    )
    if state_deriv_fpath is None:
        state_deriv_fpath = deriv_path / state_deriv_fname
    if state_deriv_fpath.exists() and not overwrite:
        print(f"Found state deriv fpath: {state_deriv_fpath}")
        state_deriv = read_derivative_npy(state_deriv_fpath)
    else:
        state_deriv = state_lds_derivative(
            raw, reference=reference, n_jobs=-1, **sysid_params
        )
        state_deriv.save(state_deriv_fpath, overwrite=overwrite)

    # if column derivative exists, and not overwrite skip
    colperturb_deriv_fpath = match_derivative(
        deriv_path, colperturb_deriv_fname, ".json", verbose=verbose
    )
    if colperturb_deriv_fpath is None:
        colperturb_deriv_fpath = deriv_path / colperturb_deriv_fname
    if colperturb_deriv_fpath.exists() and not overwrite:
        print(f"Found col deriv fpath: {colperturb_deriv_fpath}")
        cperturb_deriv = read_derivative_npy(colperturb_deriv_fpath)
    else:
        cperturb_deriv, cdeltavecs_deriv = state_perturbation_derivative(
            state_deriv, n_jobs=-1, **perturbation_params
        )
        cperturb_deriv.save(colperturb_deriv_fpath, overwrite=overwrite)

        coldeltavecs_deriv_fpath = _add_desc_to_bids_fname(
            colperturb_deriv_fpath, DERIVATIVETYPES.DELTAVECS_MATRIX.value
        )
        cdeltavecs_deriv.save(coldeltavecs_deriv_fpath, overwrite=overwrite)

    # if row
    # rowperturb_deriv_fpath = match_derivative(
    #     deriv_path, rowperturb_deriv_fname, ".json", verbose=verbose
    # )
    # if rowperturb_deriv_fpath is None:
    #     rowperturb_deriv_fpath = deriv_path / rowperturb_deriv_fname
    # if rowperturb_deriv_fpath.exists() and not overwrite:
    #     rperturb_deriv = read_derivative_npy(rowperturb_deriv_fpath)
    # else:
    #     perturbation_params["perturb_type"] = "R"
    #     rperturb_deriv, rdeltavecs_deriv = state_perturbation_derivative(
    #         state_deriv, n_jobs=-1, **perturbation_params
    #     )
    #     rperturb_deriv.save(rowperturb_deriv_fpath, overwrite=overwrite)

    #     # rowperturb_deriv_fpath
    #     rowdeltavecs_deriv_fpath = _add_desc_to_bids_fname(
    #         rowperturb_deriv_fpath, DERIVATIVETYPES.ROWDELTAVECS_MATRIX.value
    #     )
    #     rdeltavecs_deriv.save(rowdeltavecs_deriv_fpath, overwrite=overwrite)

    # plot heatmaps
    if figures_path is not None:
        figures_path.mkdir(exist_ok=True, parents=True)

        cperturb_deriv.load_data()

        # normalize derivative
        cperturb_deriv.normalize()
        # rperturb_deriv.normalize()

        # read clinical data sheet
        if excel_fpath is not None:
            # read in the dataframe of clinical datasheet
            pat_dict = read_clinical_excel(excel_fpath, subject=subject)
            # extract the SOZ channels
            soz_chs = pat_dict[ClinicalContactColumns.SOZ_CONTACTS.value]
            epz_chs = pat_dict[ClinicalContactColumns.SPREAD_CONTACTS.value]
            rz_chs = pat_dict[ClinicalContactColumns.RESECTED_CONTACTS.value]
        else:
            soz_chs = None

        cfig_basename = colperturb_deriv_fpath.with_suffix(".pdf").name
        # rfig_basename = rowperturb_deriv_fpath.with_suffix(".pdf").name

        # plot
        cperturb_deriv.plot_heatmap(
            soz_chs=soz_chs,
            cbarlabel="Fragility",
            cmap="turbo",
            # soz_chs=soz_chs,
            # figsize=(10, 8),
            # fontsize=12,
            # vmax=0.8,
            title=fig_basename,
            figure_fpath=(figures_path / cfig_basename),
        )
        # rperturb_deriv.plot_heatmap(
        #     soz_chs=soz_chs,
        #     cbarlabel="Fragility",
        #     cmap="turbo",
        #     # soz_chs=soz_chs,
        #     # figsize=(10, 8),
        #     # fontsize=12,
        #     # vmax=0.8,
        #     title=fig_basename,
        #     figure_fpath=(figures_path / rfig_basename),
        # )


if __name__ == "__main__":
    # bids root to write BIDS data to
    # the root of the BIDS dataset
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_interictal/")
    output_dir = root / "derivatives" / "originalsampling"

    figures_dir = output_dir / "figures"

    # path to excel layout file - would be changed to the datasheet locally
    excel_fpath = root / "sourcedata/ieeg_database_all.xlsx"

    # define BIDS entities
    SUBJECTS = []

    session = "extraoperative"  # only one session
    # task = "interictal"
    datatype = "ieeg"
    suffix = "ieeg"
    extension = ".edf"

    reference = "monopolar"
    sfreq = None  # either resample or don't
    overwrite = False
    verbose = True
    task = "interictal"

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
                suffix=datatype,
            )
            print(f"Analyzing {bids_path}")

            deriv_bids_path = bids_path.copy().update(check=False, extension=".npy")
            state_deriv_fname = _add_desc_to_bids_fname(
                deriv_bids_path.basename, DERIVATIVETYPES.STATE_MATRIX.value
            )
            colperturb_deriv_fname = _add_desc_to_bids_fname(
                deriv_bids_path.basename, DERIVATIVETYPES.COLPERTURB_MATRIX.value
            )
            # rowperturb_deriv_fname = _add_desc_to_bids_fname(
            #     bids_path.basename, DERIVATIVETYPES.ROWPERTURB_MATRIX.value
            # )
            run_analysis(
                bids_path,
                state_deriv_fname,
                colperturb_deriv_fname,
                # rowperturb_deriv_fname,
                reference=reference,
                resample_sfreq=sfreq,
                deriv_root=output_dir,
                verbose=verbose,
                overwrite=overwrite,
            )
            # run_fragility_analysis(bids_path, reference=reference,
            #                        resample_sfreq=sfreq,
            #                        deriv_path=output_dir,
            #                        figures_path=figures_dir,
            #                        excel_fpath=excel_fpath,
            #                        overwrite=True
            #                        )
