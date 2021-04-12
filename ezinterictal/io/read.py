from eztrack.io import read_derivative_npy
from eztrack import preprocess_ieeg
from mne_bids import get_entity_vals, read_raw_bids
from pathlib import Path


def load_data(bids_path, resample_sfreq, figure_fpath=None, verbose=None):
    """Load iEEG data in, preprocess and plot."""
    # load in the data
    raw = read_raw_bids(bids_path)

    # get only the data we care about
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False, exclude=[])

    # resample data
    if resample_sfreq:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    # load data to RAM
    raw.load_data()

    # pre-process the data using preprocess pipeline
    print("Power Line frequency is : ", raw.info["line_freq"])
    l_freq = 0.5
    h_freq = 300
    raw = preprocess_ieeg(raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    if figure_fpath is not None:
        # plot raw data
        Path(figure_fpath).parent.mkdir(exist_ok=True, parents=True)
        scale = 200e-6
        fig = raw.plot(
            decim=40,
            duration=20,
            scalings={"ecog": scale, "seeg": scale},
            n_channels=len(raw.ch_names),
            clipping=None,
        )
        fig.savefig(figure_fpath)
    return raw


def load_all_interictal_derivatives(
    deriv_root,
    datatype="ieeg",
    desc="perturbmatrix",
    ignore_subjects=None,
    verbose=True,
):
    """Load all interictal derivative fragility data into a list."""
    deriv_path = deriv_root
    all_subjects = get_entity_vals(
        deriv_path, "subject", ignore_subjects=ignore_subjects
    )
    # analyze all subjects
    subjects = all_subjects

    if verbose:
        print(f"Searching {deriv_path} for {subjects}")

    results = []
    subject_groups = []
    for subject in subjects:
        subj_deriv_path = deriv_path / f"sub-{subject}"

        if verbose:
            print(f"Looking at subject path: {subj_deriv_path}")

        # get all filepaths for that derivative description
        fpaths = subj_deriv_path.glob(f"*{desc}_{datatype}.npy")

        # load in each fpath
        for fpath in fpaths:
            # read in the dataset
            result = read_derivative_npy(fpath)

            results.append(result)
            subject_groups.append(subject)
    return results, subject_groups
