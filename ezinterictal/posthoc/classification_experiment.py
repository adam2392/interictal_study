import collections
import json
from pathlib import Path

import numpy as np
from eztrack.base.publication.study import (
    determine_feature_importances,
    format_supervised_dataset,
    tune_hyperparameters,
)
from eztrack.io import read_clinical_excel
from eztrack.utils import NumpyEncoder
from mne_bids.path import get_entity_vals
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OrdinalEncoder

from ezinterictal.io.read import load_all_interictal_derivatives
from ezinterictal.posthoc.sampling import sample_cv_clinical_complexity
from ezinterictal.posthoc.utils import format_supervised_dataset, combine_patient_predictions

# set seed and randomness for downstream reproducibility
seed = 12345
np.random.seed(seed)

random_state = 12345

max_depth = [None, 5, 10]
max_features = ["auto", "log2"]

# initialize the classifier
model_params = {
    "n_estimators": 500,
    "max_depth": max_depth[0],
    "max_features": max_features[0],
    "n_jobs": -1,
    "random_state": random_state,
}


def _sequential_aggregation(mat_list, ch_names_list, sozinds_list, outcome):
    X = []
    y = []
    agg_sozinds_list = []
    for mat, ch_names, sozinds in zip(mat_list, ch_names_list, sozinds_list):
        y.append(outcome)
        X.append(mat)
        agg_sozinds_list.append(sozinds)
    return X, y, sozinds_list


def run_classification_exp(derivatives, subjects, excel_fpath,
                           cv_inds_fpaths, study_path, clf_type='rf'):
    from rerf.rerfClassifier import rerfClassifier

    # from rerf.urerf import UnsupervisedRandomForest

    # define hyperparameters
    thresholds = [
        # None,
        # 0.1, 0.2, 0.3, 0.4,
        0.5,
        0.6,
        0.7,
        # 0.8, 0.9,
        # 0.4, 0.5, 0.6, 0.7,
        #           0.75
    ]

    metric = "roc_auc"
    # initialize the classifier
    model_params = {
        "n_estimators": 500,
        "max_depth": max_depth[0],
        "max_features": max_features[0],
        "n_jobs": -1,
        "random_state": random_state,
    }

    # initialize the type of classification function we'll use
    IMAGE_HEIGHT = 20
    if clf_type == "rf":
        clf_func = RandomForestClassifier
    elif clf_type == "srerf":
        model_params.update(
            {
                "projection_matrix": "S-RerF",
                "image_height": IMAGE_HEIGHT,
                "patch_height_max": 4,
                "patch_height_min": 1,
                "patch_width_max": 8,
                "patch_width_min": 1,
            }
        )
        clf_func = rerfClassifier
    elif clf_type == "mtmorf":
        model_params.update(
            {
                "projection_matrix": "MT-MORF",
                "image_height": IMAGE_HEIGHT,
                "patch_height_max": 4,
                "patch_height_min": 1,
                "patch_width_max": 8,
                "patch_width_min": 1,
            }
        )
        clf_func = rerfClassifier

    # 1. First load in dataset into X,y,group tuples
    X = []
    y = []
    groups = []
    sozinds_list = []
    for subject, derivative in zip(subjects, derivatives):
        # read in Excel database
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        soz_chs = pat_dict["SOZ_CONTACTS"]
        #     soz_chs = pat_dict['RESECTED_CONTACTS']
        outcome = pat_dict["OUTCOME"]

        # skip dataset if no surgery
        if outcome == "NR":
            continue

        ch_names = derivative.ch_names

        # get a nested list of all the SOZ channel indices
        _sozinds_list = [ind for ind, ch in enumerate(ch_names) if ch in soz_chs]

        # create pair X,y dataset of all subjects
        X.append(derivative.get_data())
        y.append(outcome)
        groups.append(subject)
        sozinds_list.append(_sozinds_list)
    # get the dataset parameters loaded in
    dataset_params = {"sozinds_list": sozinds_list}

    # format supervised learning datasets
    # define preprocessing to convert labels/groups into numbers
    enc = OrdinalEncoder()  # handle_unknown='ignore', sparse=False
    #     subject_groups = enc.fit_transform(np.array(subjects)[:, np.newaxis])
    y = enc.fit_transform(np.array(y)[:, np.newaxis])
    subject_groups = np.array(groups)
    # store the cross validation nested scores per feature
    nested_scores = collections.defaultdict(list)

    # 2. loop over CV indices
    for jdx in range(0, 1):
        # load in indices
        with np.load(
                cv_inds_fpaths[jdx],
                allow_pickle=True,
        ) as data_dict:
            train_pats, test_pats = data_dict["train_pats"], data_dict["test_pats"]

        # set train indices based on which subjects
        train_inds = [
            idx for idx, sub in enumerate(subjects) if sub in train_pats
        ]
        test_inds = [idx for idx, sub in enumerate(subjects) if sub in test_pats]

        # create an iterator of all possible hyperparameters
        hyperparameters = thresholds

        # run a hyperparameter tune on this subset
        master_scores = tune_hyperparameters(
            clf_func,
            unformatted_X=X.copy(),
            y=y.copy(),
            groups=subject_groups.copy(),
            train_inds=train_inds.copy(),
            test_inds=test_inds.copy(),
            hyperparameters=hyperparameters,
            dataset_params=dataset_params,
            **model_params,
        )
        print("Done tuning data hyperparameters...")

        # 3. evaluate on test set
        # get the best classifier based on pre-chosen metric
        test_key = f"test_{metric}"
        metric_list = [np.mean(scores[test_key]) for scores in master_scores]
        best_index = np.argmax(metric_list)

        # get the best estimator within that inner cv
        best_metric_ind = np.argmax(master_scores[best_index]["test_roc_auc"])
        best_estimator = master_scores[best_index]["estimator"][best_metric_ind]
        best_threshold = master_scores[best_index]["hyperparameters"]

        # format supervised dataset
        X_formatted, dropped_inds = format_supervised_dataset(
            X.copy(),
            **dataset_params,
            threshold=best_threshold,
        )

        # evaluate on the testing dataset
        X_test, y_test = np.array(X_formatted)[test_inds, ...], np.array(y)[test_inds]
        groups_test = np.array(subject_groups)[test_inds]

        # resample the held-out test data via bootstrap
        test_sozinds_list = np.asarray(dataset_params["sozinds_list"])[test_inds]
        test_onsetwin_list = np.asarray(dataset_params["onsetwin_list"])[test_inds]

        X_boot, y_boot = X_test.copy(), y_test.copy()

        y_pred_prob = best_estimator.predict_proba(X_boot)[:, 1]
        y_pred = best_estimator.predict(X_boot)

        # store analysis done on the validation group
        nested_scores["validate_groups"].append(groups_test)
        nested_scores["validate_subjects"].append(groups_test)
        nested_scores["hyperparameters"].append(best_threshold)

        if clf_type == "rf":
            # pop estimator
            nested_scores["estimator"].append(best_estimator)
        estimators.append(best_estimator)

        # store the actual outcomes and the predicted probabilities
        nested_scores["validate_ytrue"].append(list(y_test))
        nested_scores["validate_ypred_prob"].append(list(y_pred_prob))
        nested_scores["validate_ypred"].append(list(y_pred))

        # store ROC curve metrics on the held-out test set
        fpr, tpr, thresholds = roc_curve(y_boot, y_pred_prob, pos_label=1)
        fnr, tnr, neg_thresholds = roc_curve(y_boot, y_pred_prob, pos_label=0)
        nested_scores["validate_fpr"].append(list(fpr))
        nested_scores["validate_tpr"].append(list(tpr))
        nested_scores["validate_fnr"].append(list(fnr))
        nested_scores["validate_tnr"].append(list(tnr))
        nested_scores["validate_thresholds"].append(list(thresholds))

        print("Done analyzing ROC stats...")

        # run the feature importances
        # compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_boot, y_pred_prob, n_bins=10, strategy="quantile"
        )
        clf_brier_score = np.round(
            brier_score_loss(y_boot, y_pred_prob, pos_label=np.array(y_boot).max()), 2
        )

        print("Done analyzing calibration stats...")

        # store ingredients for a calibration curve
        nested_scores["validate_brier_score"].append(float(clf_brier_score))
        nested_scores["validate_fraction_pos"].append(list(fraction_of_positives))
        nested_scores["validate_mean_pred_value"].append(list(mean_predicted_value))

        # store outputs to run McNemars test and Cochrans Q test
        # get the shape of a single feature "vector" / structure array
        pat_predictions, pat_true = combine_patient_predictions(
            y_boot, y_pred_prob, groups_test
        )
        nested_scores["validate_pat_predictions"].append(pat_predictions)
        nested_scores["validate_pat_true"].append(pat_true)

        # store output for feature importances
        X_shape = X_boot[0].reshape((IMAGE_HEIGHT, -1)).shape
        if clf_type == "rf":
            n_jobs = -1
        else:
            n_jobs = 1
        results = determine_feature_importances(
            best_estimator, X_boot, y_boot, n_jobs=n_jobs
        )
        imp_std = results.importances_std
        imp_vals = results.importances_mean
        nested_scores["validate_imp_mean"].append(list(imp_vals))
        nested_scores["validate_imp_std"].append(list(imp_std))

        print("Done analyzing feature importances...")

        # save intermediate analyses
        clf_func_path = (
                study_path / "clf" / f"{clf_type}_classifiers_{feature_name}_{jdx}.npz"
        )
        clf_func_path.parent.mkdir(exist_ok=True, parents=True)

        # nested CV scores
        nested_scores_fpath = (
                study_path / f"study_nested_scores_{clf_type}_{feature_name}_{jdx}.json"
        )

        # save the estimators
        if clf_type not in ["srerf", "mtmorf"]:
            estimators = nested_scores.pop("estimator")
            np.savez_compressed(clf_func_path, estimators=estimators)

        # save all the master scores as a JSON file
        with open(nested_scores_fpath, "w") as fin:
            json.dump(nested_scores, fin, cls=NumpyEncoder)

        del master_scores
        del estimators
        del best_estimator


if __name__ == "__main__":
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        # the root of the BIDS dataset
        root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
        deriv_root = root / 'derivatives' / 'interictal'

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
        deriv_root = Path("/home/adam2392/hdd/epilepsy_bids") / 'derivatives' / 'interictal'

    feature_names = [
        "fragility",
    ]
    sfreq = 1000
    reference = 'average'
    datatype = 'ieeg'
    description = 'perturbmatrix'
    clf_type = "mtmorf"
    train_size = 0.5

    for feature_name in feature_names:
        # create the derivatives chain
        deriv_chain = Path(f'{sfreq}Hz') / feature_name / reference
        deriv_path = deriv_root / deriv_chain
        subjects = get_entity_vals(root, 'subject',
                                   ignore_tasks=['ictal'],
                                   ignore_datatypes=['eeg'])

        print('Found these subjects to run cv over...')
        print(f'Total subjects {len(subjects)}')

        cv_ind_fpaths = sample_cv_clinical_complexity(
            subjects, deriv_path, excel_fpath, train_size=train_size, n_splits=10
        )

    for feature_name in feature_names:
        # load all datasets
        interictal_results, subject_groups = load_all_interictal_derivatives(deriv_root=deriv_root,
                                                                             datatype=datatype,
                                                                             desc=description)

        run_classification_exp(interictal_results,
                               subject_groups, excel_fpath, cv_ind_fpaths,
                               study_path=deriv_root / 'study',
                               clf_type=clf_type)
