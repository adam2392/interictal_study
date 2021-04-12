from pathlib import Path

from eztrack.fragility.fragility import (
    state_perturbation_array,
    state_perturbation_derivative,
)
from eztrack.io import read_derivative_npy
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from mne.utils import warn
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals


def run_analysis(
    bids_path,
    reference="monopolar",
    resample_sfreq=None,
    deriv_path=None,
    figures_path=None,
    excel_fpath=None,
    verbose=True,
    overwrite=False,
):
    subject = bids_path.subject
    datatype = bids_path.datatype

    # load in the data
    raw = read_raw_bids(bids_path)
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False)

    if resample_sfreq:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    if deriv_path is None:
        deriv_path = bids_path.root / "derivatives"
    deriv_path = (
        deriv_path
        / f"{int(raw.info['sfreq'])}Hz"
        / "fragility"
        / reference
        / f"sub-{subject}"
    )
    # set where to save the data output to
    if figures_path is None:
        figures_path = bids_path.root / "derivatives" / "figures"
    figures_path = (
        figures_path
        / f"{int(raw.info['sfreq'])}Hz"
        / "fragility"
        / reference
        / f"sub-{subject}"
    )

    # use the same basename to save the data
    deriv_basename = bids_path.basename
    deriv_basename = _add_desc_to_bids_fname(
        deriv_basename,
        description=DERIVATIVETYPES.ROWPERTURB_MATRIX.value,
        verbose=False,
    )
    deriv_fpath = deriv_path / deriv_basename
    if deriv_fpath.exists() and not overwrite:
        warn(f"The {deriv_basename}.npy exists, but overwrite if False.")
        return

    # write results to
    source_entities = bids_path.entities
    raw_basename = BIDSPath(**source_entities).basename
    deriv_fname = list(deriv_path.glob(f"{raw_basename}*rowperturbmatrix*"))
    if len(deriv_fname) > 0:
        deriv_fname = deriv_fname[0]

    deriv_fname = list(deriv_path.glob(f"{raw_basename}*statematrix*"))[0]
    state_arr, result_info, metadata = read_result_eztrack(
        deriv_fname=deriv_fname, description="statematrix", normalize=False
    )
    # run row perturbation
    radius = 1.5
    perturb_type = "R"
    pert_mats, delta_vecs_arr = state_perturbation_array(
        state_arr, radius=radius, perturb_type=perturb_type, n_jobs=-1
    )

    # write results to
    pert_sidecar = write_result_array(
        pert_mats,
        metadata=metadata,
        deriv_fpath=deriv_fpath,
        verbose=verbose,
    )
    result = Result(pert_mats, raw.info, metadata=metadata)

    # deriv_fname = deriv_path / bids_path.basename
    fig_basename = deriv_basename
    # result = read_result_eztrack(deriv_fname=deriv_fname,
    #                              description='rowperturbmatrix',
    #                              normalize=False)

    result.normalize()
    # create the heatmap
    plot_result_heatmap(
        result=result,
        fig_basename=fig_basename,
        figures_path=figures_path,
        excel_fpath=excel_fpath,
        show_soz=True,
    )


def run_row_analysis(root, deriv_root, deriv_chain):
    """Computes the row perturbation fragility analysis for dataset."""
    deriv_path = deriv_root / deriv_chain
    deriv_subjects = get_entity_vals(deriv_path, "subject")
    raw_subjects = get_entity_vals(root, "subject")

    session = "presurgery"
    task = "interictal"
    suffix = "ieeg"
    datatype = "ieeg"
    extension = ".vhdr"
    overwrite = False

    # derivative descriptions
    description = "statematrix"
    row_description = "rowperturbmatrix"
    row_delta_description = "rowdeltamatrix"

    # row perturbation parameters
    model_params = {
        "radius": 1.5,
        "perturb_type": "R",
    }

    # get the raw data file
    for raw_sub in raw_subjects:
        # does not have any subjects in derivative already
        if raw_sub not in deriv_subjects:
            continue

        raw_bids_path = BIDSPath(
            subject=raw_sub,
            session=session,
            task=task,
            suffix=suffix,
            datatype=datatype,
            extension=extension,
            root=root,
        )

        # get all the raw filenames
        raw_fnames = raw_bids_path.match()
        for raw_path in raw_fnames:
            # get the source bids names
            source_entities = raw_path.entities

            # get the derivative path
            deriv_bids_path = BIDSPath(
                suffix=f"desc-{description}_ieeg",
                extension=".npy",
                check=False,
                **source_entities,
            )

            # get the desired output derivative path
            row_deriv_path = deriv_bids_path.copy().update(
                suffix=f"desc-{row_description}_ieeg",
            )
            row_output_fpath = deriv_path / f"sub-{raw_sub}" / row_deriv_path.basename
            state_fpath = deriv_path / f"sub-{raw_sub}" / deriv_bids_path.basename

            # check if we want to overwrite the file or not
            if not overwrite and row_output_fpath.exists():
                continue

            # get the corresponding state matrix file
            state_deriv = read_derivative_npy(state_fpath)

            # compute row fragility
            print(
                f"Computing row fragility for {state_deriv} to "
                f"write to {row_output_fpath}"
            )
            pertnorm_deriv, deltavecs_deriv = state_perturbation_derivative(
                state_deriv, **model_params
            )

            # save the output
            pertnorm_deriv.save(row_output_fpath)

            deltavecs_row_fpath = row_output_fpath.as_posix().replace(
                row_description, row_delta_description
            )
            deltavecs_deriv.save(deltavecs_row_fpath)


def main():
    sampling_strategy = "originalsampling"
    reference = "average"

    # path directories
    root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
    deriv_root = root / "derivatives"
    deriv_chain = Path("interictal") / sampling_strategy / "fragility" / reference

    run_row_analysis(root, deriv_root, deriv_chain)


if __name__ == "__main__":
    main()
    exit(1)

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
        output_dir = Path("/home/adam2392/hdd2") / "derivatives" / "interictal"

        # figures directory
        figures_dir = output_dir / "figures"

    # define BIDS entities
    # SUBJECTS = [
    #     # 'pt1', 'pt2', 'pt3',  # NIH
    #     'jh103', 'jh105',  # JHH
    #     # 'umf001', 'umf002', 'umf003', 'umf005', # UMF
    # ]

    session = "presurgery"  # only one session
    task = "interictal"
    datatype = "ieeg"
    acquisition = "ecog"  # or SEEG
    extension = ".vhdr"

    if acquisition == "ecog":
        ignore_acquisitions = ["seeg"]
    elif acquisition == "seeg":
        ignore_acquisitions = ["ecog"]

    # analysis parameters
    reference = "monopolar"
    sfreq = None  # either resample or don't

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")

    for subject in all_subjects:
        # if subject not in SUBJECTS:
        #     continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]
        all_tasks = get_entity_vals(root, "task", ignore_subjects=ignore_subs)
        ignore_tasks = [tsk for tsk in all_tasks if tsk != task]

        print(f"Analyzing {task} task for {subject}.")
        ignore_tasks = [tsk for tsk in all_tasks if tsk != task]
        runs = get_entity_vals(
            root,
            "run",
            ignore_subjects=ignore_subs,
            ignore_tasks=ignore_tasks,
            ignore_acquisitions=ignore_acquisitions,
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
                acquisition=acquisition,
                suffix=datatype,
                root=root,
                extension=extension,
            )
            print(f"Analyzing {bids_path}")

            run_analysis(
                bids_path,
                reference=reference,
                resample_sfreq=sfreq,
                deriv_path=output_dir,
                figures_path=figures_dir,
                excel_fpath=excel_fpath,
                overwrite=False,
            )
