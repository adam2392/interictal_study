import numpy as np
from eztrack import read_clinical_excel

from ezinterictal.posthoc.config import random_state


def sample_cv_clinical_complexity(
    subjects, study_path, excel_fpath, train_size=0.5, n_splits=10
):
    from sklearn.model_selection import StratifiedShuffleSplit

    # create held-out test dataset
    # create separate pool of subjects for testing dataset
    # 1. Cross Validation Training / Testing Split
    # subjects = np.unique(subject_groups)

    clinical_complexity = []
    pat_df = read_clinical_excel(excel_fpath, keep_as_df=True)
    for subj in subjects:
        pat_row = pat_df[pat_df["PATIENT_ID"] == subj.upper()]
        if pat_row.size == 0:
            print(f"did not find {subj}")
            continue
        clinical_complexity.append(pat_row["CLINICAL_COMPLEXITY"].values[0])
    clinical_complexity = np.array(clinical_complexity).astype(float)

    gss = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=train_size, random_state=random_state
    )
    fpaths = []
    for jdx, (train_inds, test_inds) in enumerate(
        gss.split(subjects, clinical_complexity)
    ):
        train_pats = np.array(subjects)[train_inds]
        test_pats = np.array(subjects)[test_inds]
        fpath = study_path / f"{jdx}-inds.npz"
        np.savez_compressed(
            fpath,
            train_pats=train_pats,
            test_pats=test_pats,
        )
        print(f"train patients: {train_pats}")
        print(f"test patients: {test_pats}")
        fpaths.append(fpath.as_posix())
    return fpaths
