"""
This script extracts datasets into a `.npz` file in order to facilitate faster
IO for the results leading to the figures in the manuscript.
"""
import time
from pathlib import Path

import mne

from ezinterictal.io.read import load_all_interictal_derivatives

mne.set_log_level("ERROR")


def save_trimmed_slices(result, trim_length, deriv_path, start=10, verbose=True):
    # hack to accomodate earlier derivatives computed
    if result.derivative_sfreq is None:
        result.info['derivative_sfreq'] = result.info['sfreq'] / 125
        result._update_times()
        print(f'reset derivative sfreq to {result.derivative_sfreq}')

    if verbose:
        print(f'The original length of the dataset {result} is {len(result)}')

    # trim_length is in seconds
    result = result.crop(tmin=start, tmax=trim_length + start)

    if verbose:
        print(f'The new length of the dataset {result} is {len(result)}')

    # perform save
    fname = Path(result.filenames[0]).name
    fpath = Path(deriv_path) / fname

    # save the output file sliced
    result.save(fpath, overwrite=True)


if __name__ == '__main__':
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        # the root of the BIDS dataset
        root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
        output_dir = root / 'derivatives' / 'interictal'

        figures_dir = output_dir / 'figures'

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
        output_dir = Path("/home/adam2392/hdd/epilepsy_bids") / 'derivatives' / 'interictal'

        # figures directory
        figures_dir = output_dir / 'figures'

    # define various list's of patients
    separate_pats = [
        "la09",
        "la27",
        "la29",
        "nl02",
        "pt11",
        "tvb7",
        "tvb18",
        "jh107",
    ]

    # BIDS entities
    session = "presurgery"
    acquisition = "seeg"
    task = "interictal"
    datatype = "ieeg"
    reference = "monopolar"
    patient_aggregation_method = None

    sfreq = 1000

    # to perform the experiment
    expname = "sliced"
    featuremodels = [
        "fragility",
    ]

    descriptions = [
        'perturbmatrix'
    ]

    centers = [
        "nih",
        "jhu",
        "umf",
        "clevelandtvb",
        "clevelandnl",
        "cleveland",
    ]

    # create sub-directories
    # explore trimmed lengths of 5, 10, 30, 60 seconds
    trim_length = 5
    pub_data_dir = output_dir / 'publication_data_sliced' / f'{trim_length}secs'
    pub_data_dir.mkdir(parents=True, exist_ok=True)

    # save intermediate results cropped for each dataset
    for feature_name in featuremodels:
        start = time.process_time()
        interictal_results = load_all_interictal_derivatives(deriv_root=output_dir,
                                                             sfreq=sfreq, reference=reference,
                                                             datatype=datatype,
                                                             feature_name=feature_name,
                                                             desc=descriptions[0])
        end = time.process_time()

        print(f'LOADING ALL DATASETS TOOK {end - start} time.')
        print(len(interictal_results))

        for result in interictal_results:
            save_trimmed_slices(result, trim_length=trim_length, deriv_path=pub_data_dir,
                                start=10)

        # get the (X, y) tuple pairs
        # unformatted_X, y, sozinds_list, onsetwin_list, subject_groups = extract_Xy_pairs(
        #     feature_subject_dict,
        #     excel_fpath=excel_fpath,
        #     patient_aggregation_method=patient_aggregation_method,
        #     verbose=False,
        # )
        # # print(unformatted_X[0])
        # # break
        # print(
        #     len(unformatted_X),
        #     len(y),
        #     len(subject_groups),
        #     len(onsetwin_list),
        #     len(sozinds_list),
        # )
        # fpath = (
        #         Path(deriv_path).parent / "baselinesliced" / f"{feature_name}_unformatted.npz"
        # )
        # fpath.parent.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed(
        #     fpath,
        #     unformatted_X=unformatted_X,
        #     y=y,
        #     sozinds_list=sozinds_list,
        #     onsetwin_list=onsetwin_list,
        #     subject_groups=subject_groups,
        #     subjects=subjects,
        # )
