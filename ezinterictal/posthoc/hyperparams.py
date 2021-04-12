import numpy as np
from rerf.rerfClassifier import rerfClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.model_selection import GroupKFold, cross_validate

from ezinterictal.posthoc.utils import format_supervised_dataset


def _evaluate_model(
    clf_func,
    model_params,
    train_inds,
    X_formatted,
    y,
    groups,
    cv,
    dropped_inds=None,
):
    y = np.array(y).copy()
    groups = np.array(groups).copy()
    train_inds = train_inds.copy()

    # if dropped_inds:
    #     for ind in dropped_inds:
    #         # if ind in train_inds:
    #         where_ind = np.where(train_inds >= ind)[0]
    #         train_inds[where_ind] -= 1
    #         train_inds = train_inds[:-1]
    #         # delete index in y, groups
    #         y = np.delete(y, ind)
    #         groups = np.delete(groups, ind)

    # instantiate model
    clf = clf_func(**model_params)

    # note that training data (Xtrain, ytrain) will get split again
    Xtrain, ytrain = X_formatted[train_inds, ...], y[train_inds]
    groups_train = groups[train_inds]

    # perform CV using Sklearn
    scoring_funcs = {
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "average_precision": average_precision_score,
    }
    scores = cross_validate(
        clf,
        Xtrain,
        ytrain,
        groups=groups_train,
        cv=cv,
        scoring=list(scoring_funcs.keys()),
        return_estimator=True,
        return_train_score=True,
    )

    return scores


def tune_hyperparameters(
    clf_func,
    unformatted_X,
    y,
    groups,
    train_inds,
    hyperparameters,
    dataset_params,
    **model_params,
):
    """Perform hyperparameter tuning.

    Pass in X and y dataset that are unformatted yet, and then follow
    a data pipeline that:

    - create the formatted dataset
    - applies hyperparameters
    - cross-validate

    Parameters
    ----------
    clf_func :
    unformatted_X :
    y :
    groups :
    train_inds :
    test_inds :
    hyperparameters :
    dataset_params :
    model_params :

    Returns
    -------
    master_scores: dict
    """

    # CV object to perform training of classifier
    # create Grouped Folds to estimate the mean +/- std performancee
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)

    # track all cross validation score dictionaries
    master_scores = []

    print(f"Using classifier: {clf_func}")
    for idx, hyperparam in enumerate(hyperparameters):
        # extract the hyperparameter explicitly
        threshold = hyperparam
        hyperparam_str = f"threshold-{threshold}"
        # apply the hyperparameters to the data
        #         print(unformatted_X.shape)
        X_formatted, dropped_inds = format_supervised_dataset(
            unformatted_X,
            **dataset_params,
            threshold=threshold,
            clf_type=model_params.get("projection_matrix"),
        )
        if idx == 0:
            print("The formatted dataset is X")
            print(X_formatted.shape)

        scores = _evaluate_model(
            clf_func,
            model_params,
            train_inds,
            X_formatted,
            y,
            groups,
            cv,
            dropped_inds=dropped_inds,
        )
        # get the best classifier based on pre-chosen metric
        scores["hyperparameters"] = hyperparam

        master_scores.append(scores)

    return master_scores
