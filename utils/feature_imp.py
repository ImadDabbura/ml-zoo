import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


def oob_accuracy(rf_clf, X_train, y_train):
    """Compute the out-of-bag accuracy of random forest classifier"""
    n_samples = X_train.shape[0]
    n_classes = len(np.bincount(y_train))
    n_preds = np.zeros((n_samples))
    preds_matrix = np.zeros((n_samples, n_classes))

    # Iterate over all trees
    for tree in rf_clf.estimators_:
        # Generate unsampled indices
        unsampled_idxs = _generate_unsampled_indices(
            tree.random_state, n_samples)
        preds = tree.predict_proba(X_train[unsampled_idxs, :])
        preds_matrix[unsampled_idxs, :] += preds
        n_preds[unsampled_idxs] += 1

    # Avoid dividing by zero if some samples weren't included
    if (n_preds == 0).any():
        warnings.warn("Some features didn't have OOB samples.")
        y_train = y_train[n_preds != 0]
        preds_matrix = preds_matrix[n_preds != 0, :]

    preds_classes = np.argmax(preds_matrix, axis=1)
    oob_score = (y_train == preds_classes).mean()

    return oob_score


def oob_r2(rf_reg, X_train, y_train):
    """Compute the out-of-bag R2 of random forest regressor"""
    n_samples = X_train.shape[0]
    n_preds = np.zeros(n_samples)
    preds_matrix = np.zeros((n_samples))

    # Iterate over all trees
    for tree in rf_reg.estimators_:
        # Generate unsampled indices
        unsampled_idxs = _generate_unsampled_indices(
            tree.random_state, n_samples)
        preds = tree.predict(X_train[unsampled_idxs, :])
        preds_matrix[unsampled_idxs] += preds
        n_preds[unsampled_idxs] += 1

    # Avoid dividing by zero if some samples weren't included
    if (n_preds == 0).any():
        warnings.warn("Some features didn't have OOB samples.")
        # Discard samples weren't OOB in any feature
        y_train = y_train[n_preds != 0]
        preds_matrix = preds_matrix[n_preds != 0]
    avg_preds = preds_matrix / n_preds
    oob_score = r2_score(y_train, avg_preds, )
    return oob_score


def permutation_importances(rf_clf, X_train, y_train, scorer):
    """"
        Compute feature importances using permutation based on scorer OOB
        metric.
    """
    feat_imp = []
    base_score = scorer(rf_clf, X_train, y_train)

    for j in range(X_train.shape[1]):
        temp = X_train[:, j].copy()
        X_train[:, j] = np.random.permutation(X_train[:, j])
        score = scorer(rf_clf, X_train, y_train)
        feat_imp.append(base_score - score)
        X_train[:, j] = temp

    return np.array(feat_imp)


def permutation_importances_cv(rf_clf, X_train, y_train, k=3):
    """"
        Compute feature importances using permutation based on `k-fold` cross
        validation of the metric.
    """
    def scorer(model):
        score = cross_val_score(
            rf_clf, X_train, y_train, cv=k, scoring='accuracy', n_jobs=-1)
        return score.mean()

    base_score = scorer(rf_clf)
    feat_imp = []

    for j in range(X_train.shape[1]):
        temp = X_train[:, j].copy()
        X_train[:, j] = np.random.permutation(X_train[:, j])
        score = scorer(rf_clf)
        feat_imp.append(base_score - score)
        X_train[:, j] = temp

    return np.array(feat_imp)


def drop_column_importances(rf_clf, X_train, y_train):
    # Fit rf classifier with all columns to get the baseline oob score
    rf = clone(rf_clf)
    rf.random_state = 123
    rf.fit(X_train, y_train)
    base_score = rf.oob_score_
    feat_imp = []

    # Iterate over all features to compute the importance of each one
    for j in range(X_train.shape[1]):
        X = np.delete(X_train, j, axis=1)
        rf = clone(rf_clf)
        rf.random_state = 123
        rf.fit(X, y_train)
        score = rf.oob_score_
        feat_imp.append(base_score - score)

    return np.array(feat_imp)


def drop_column_importances_cv(rf_clf, X_train, y_train, k=3):
    def scorer(model, X_train):
        score = cross_val_score(
            rf_clf, X_train, y_train, cv=k, scoring='accuracy', n_jobs=-1)
        return score.mean()

    # Fit rf classifier with all columns to get the baseline oob score
    rf = clone(rf_clf)
    rf.random_state = 123
    base_score = scorer(rf, X_train)
    feat_imp = []

    # Iterate over all features to compute the importance of each one
    for j in range(X_train.shape[1]):
        X = np.delete(X_train, j, axis=1)
        rf = clone(rf_clf)
        rf.random_state = 123
        score = scorer(rf, X)
        feat_imp.append(base_score - score)

    return np.array(feat_imp)
