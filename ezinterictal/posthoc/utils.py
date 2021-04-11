import collections

import numpy as np
from eztrack.base.utils import _smooth_matrix
from eztrack.base.utils.preprocess_utils import _apply_threshold


def combine_patient_predictions(
        ytrues, ypred_probs, subject_groups, pat_predictions=None, pat_true=None
):
    if pat_predictions is None or pat_true is None:
        pat_predictions = collections.defaultdict(list)
        pat_true = dict()

    # loop through things
    for ytrue, ypred_proba, subject in zip(ytrues, ypred_probs, subject_groups):
        pat_predictions[subject].append(float(ypred_proba))

        if subject not in pat_true:
            pat_true[subject] = ytrue[0]
        else:
            if pat_true[subject] != ytrue[0]:
                raise RuntimeError("wtf subject should all match...")
    return pat_predictions, pat_true


def format_supervised_dataset(
        X,
        sozinds_list,
        threshold=None,
        smooth=None,
        clf_type=None,
):
    """Format a supervised learning dataset with (unformatted_X, y).

    This formats unformatted_X to a 4 x T dataset and y is in a set of labels (e.g. 0, or 1 for binary).

    Hyperparameters are:
        - threshold
        - weighting scheme
        - windows chosen
        - smoothing kernel over time

    Parameters
    ----------
    X :
    sozinds_list :
    threshold :
    smooth :

    Returns
    -------
    newX: np.ndarray
        Stacked data matrix with each heatmap now condensed to four sufficient statistics:
            - mean(SOZ)
            - std(SOZ)
            - mean(SOZ^C)
            - std(SOZ^C)
    """
    newX = []
    dropped_inds = []
    for idx, (data_mat, sozinds) in enumerate(
            zip(X, sozinds_list)
    ):
        if smooth is not None:
            # apply moving avg filter
            data_mat = _smooth_matrix(data_mat, window_len=8)

        if threshold is not None:
            data_mat = _apply_threshold(data_mat, threshold=threshold)

        # assemble 4-row dataset
        nsozinds = [i for i in range(data_mat.shape[0]) if i not in sozinds]
        try:
            soz_mat = data_mat[sozinds, :]
            nsoz_mat = data_mat[nsozinds, :]
        except IndexError as e:
            raise IndexError(e)

        if clf_type is not None:
            if clf_type == 'RerF':
                soz_mat = np.mean(soz_mat, axis=1)
                nsoz_mat = np.mean(nsoz_mat, axis=1)

        # new_data_mat = np.vstack(
        #     (
        #         np.mean(soz_mat, axis=0),
        #         np.std(soz_mat, axis=0),
        #         np.quantile(soz_mat, q=0.25, axis=0),
        #         np.quantile(soz_mat, q=0.5, axis=0),
        #         np.quantile(soz_mat, q=0.75, axis=0),
        #         np.mean(nsoz_mat, axis=0),
        #         np.std(nsoz_mat, axis=0),
        #         np.quantile(nsoz_mat, q=0.25, axis=0),
        #         np.quantile(nsoz_mat, q=0.5, axis=0),
        #         np.quantile(nsoz_mat, q=0.75, axis=0),
        #     )
        # )

        new_data_mat = np.vstack(
            (
                # np.mean(soz_mat, axis=0),
                # np.std(soz_mat, axis=0),
                np.quantile(soz_mat, q=0.1, axis=0),
                np.quantile(soz_mat, q=0.2, axis=0),
                np.quantile(soz_mat, q=0.3, axis=0),
                np.quantile(soz_mat, q=0.4, axis=0),
                np.quantile(soz_mat, q=0.5, axis=0),
                np.quantile(soz_mat, q=0.6, axis=0),
                np.quantile(soz_mat, q=0.7, axis=0),
                np.quantile(soz_mat, q=0.8, axis=0),
                np.quantile(soz_mat, q=0.9, axis=0),
                np.quantile(soz_mat, q=1.0, axis=0),
                np.quantile(nsoz_mat, q=0.1, axis=0),
                # np.mean(nsoz_mat, axis=0),
                # np.std(nsoz_mat, axis=0),
                np.quantile(nsoz_mat, q=0.2, axis=0),
                np.quantile(nsoz_mat, q=0.3, axis=0),
                np.quantile(nsoz_mat, q=0.4, axis=0),
                np.quantile(nsoz_mat, q=0.5, axis=0),
                np.quantile(nsoz_mat, q=0.6, axis=0),
                np.quantile(nsoz_mat, q=0.7, axis=0),
                np.quantile(nsoz_mat, q=0.8, axis=0),
                np.quantile(nsoz_mat, q=0.9, axis=0),
                np.quantile(nsoz_mat, q=1.0, axis=0),
            )
        )
        newX.append(new_data_mat.reshape(-1, 1).squeeze())

    return np.asarray(newX), dropped_inds


def determine_feature_importances(clf, X, y, n_jobs):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator=clf,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=5,
        n_jobs=n_jobs,
        random_state=1,
    )

    std = result.importances_std
    indices = np.argsort(result.importances_mean)[::-1]

    return result
