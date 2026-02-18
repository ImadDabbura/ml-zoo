from warnings import warn
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from warnings import warn

import numpy as np
import pandas as pd


FEATURES = [
    'FTE', 'Facility_or_Department', 'Function_Description', 'Fund_Description',
    'Job_Title_Description', 'Location_Description', 'Object_Description',
    'Position_Extra', 'Program_Description', 'SubFund_Description',
    'Sub_Object_Description', 'Text_1', 'Text_2', 'Text_3', 'Text_4', 'Total'
]

NUMERICAL_FEATURES = [
    'FTE', 'Total'
]

TEXT_FEATURES = [
    'Facility_or_Department', 'Function_Description', 'Fund_Description',
    'Job_Title_Description', 'Location_Description', 'Object_Description',
    'Position_Extra', 'Program_Description', 'SubFund_Description',
    'Sub_Object_Description', 'Text_1', 'Text_2', 'Text_3', 'Text_4',
]
LABELS = [
    'Function', 'Object_Type', 'Operating_Status', 'Position_Type',
    'Pre_K', 'Reporting', 'Sharing', 'Student_Type', 'Use'
]

def multilabel_sample(y, size, min_count, seed=None):
    # Check if all labels are binary
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).any():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('Labels have to be binary')
    
    # Check if we have enough examples (>= min_count) per class
    if (np.sum(y, axis=0) < min_count).any():
        raise ValueError('Some classes have less than min_count examples.')
    
    if size <=1:
        size = size * y.shape[0]

    # Check if test_size < min_count * labels
    if min_count * y.shape[1] > size:
        warn(f'test_size is less than min_count * columns')
        size = min_count * y.shape[1]
    
    # Set random_seed
    np.random.seed(seed if seed is not None else 123)

    # Check if y is a DataFrame
    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(len(y))
    
    # Instantiate sample indices array
    sample_idxs = np.array([], dtype=choices.dtype)

    # Iterare over all columns to get sample idxs
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        # Sample without replacemnet
        label_sample_idxs = np.random.choice(label_choices,
                                             min_count,
                                             replace=False)
        sample_idxs = np.concatenate([sample_idxs, label_sample_idxs])
    
    # Make sure we have unique indices
    sample_idxs = np.unique(sample_idxs)

    # Get the number needed to get the size specified
    remaining_size = int(size - len(sample_idxs))

    # Resample and add to sample_idxs to match size
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_idxs = np.random.choice(remaining_choices,
                                      remaining_size,
                                      replace=False)
    
    return np.concatenate([sample_idxs, remaining_idxs])


def multilabel_train_test_split(X, y, size, min_count, seed=None):

    # Get test indices
    test_idxs = multilabel_sample(y, size, min_count, seed)
    # Get indices for the whole datatset
    index = (y.index if isinstance(y, pd.DataFrame) else np.arange(len(y)))
    # Train indices will be all indices that are not in test indices
    train_idxs = np.setdiff1d(index, test_idxs)
    # Split the data
    if isinstance(X, pd.DataFrame):
        X_train, y_train = X.loc[train_idxs], y.loc[train_idxs]
        X_test, y_test = X.loc[test_idxs], y.loc[test_idxs]
    else:
        X_train, y_train = X[train_idxs], y.loc[train_idxs]
        X_test, y_test = X[test_idxs], y.loc[test_idxs]
    return (X_train, X_test, y_train, y_test)


def score(preds, actuals, labels_indices):
    assert (preds.columns == actuals.columns).all
    assert (preds.index == actuals.index).all()
    
    eps = 1e-15
    n = len(actuals)
    class_scores = {}
    
    for k, label_indices in labels_indices.items():
        preds_k = preds.values[:, label_indices].astype('float')
        # Normalize the preds to be between [eps, 1- eps]
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.Inf)
        preds_k = np.clip(preds_k, eps, 1- eps)
        actuals_k = actuals.values[:, label_indices]
        class_scores[k] = - (1 / n) * (actuals_k * np.log(preds_k)).sum()
    
    return class_scores, np.average(list(class_scores.values()))


def make_submission(
    estimator, test_fname, text_features, numerical_features, labels,
    title='submission'):
    test_df = pd.read_csv(test_fname, index_col=0, low_memory=False)
    text_df = combine_text_features(test_df, text_features)
    test_df = pd.concat([test_df[numerbical_features], text_df], axis=1)
    preds = estimator.predict_proba(test_df)
    preds_df = pd.DataFrame(data=preds,
                            index=test_df.index,
                            columns=labels)
    preds_df.to_csv(f'data/submissions/{title}.csv')


def combine_text_features(df, text_features, drop=True):
    # Clone df and filter out columns that are not text
    text_df = df[text_features].copy()
    # Fill NAs with blank
    text_df = text_df.fillna('', axis=0)
    # Combine all text features into one text separated by whitespace
    text_df['text'] = text_df.apply(lambda x: ' '.join(x), axis=1)
    # Drop raw text features and keep the new combined text feature
    text_df.drop(text_features, inplace=True, axis=1)
    
    return text_df
