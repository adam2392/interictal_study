from eztrack.io import read_derivative
from mne_bids import get_entity_vals
from pathlib import Path

def load_all_interictal_derivatives(deriv_root, datatype='ieeg', desc='perturbmatrix', ignore_subjects=None, verbose=True):
    """Load all interictal derivative fragility data into a list."""
    deriv_path = deriv_root
    all_subjects = get_entity_vals(deriv_path, "subject", ignore_subjects=ignore_subjects)
    # analyze all subjects
    subjects = all_subjects

    if verbose:
        print(f'Searching {deriv_path} for {subjects}')

    results = []
    subject_groups = []
    for subject in subjects:
        subj_deriv_path = deriv_path / f'sub-{subject}'

        if verbose:
            print(f'Looking at subject path: {subj_deriv_path}')

        # get all filepaths for that derivative description
        fpaths = subj_deriv_path.glob(f'*{desc}_{datatype}.npy')

        # load in each fpath
        for fpath in fpaths:
            # read in the dataset
            result = read_derivative(fpath)

            results.append(result)
            subject_groups.append(subject)
    return results, subject_groups