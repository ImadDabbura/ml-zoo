import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .feature_imp import (permutation_importances,
                          drop_column_importances,
                          oob_r2,
                          oob_accuracy)


def plot_class_dist(array, x_labels, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.countplot(array)
    plt.xticks(range(len(x_labels)), x_labels, fontsize=18)
    plt.ylabel('Count', fontdict={'fontsize': 18})
    plt.title('Classes Distribution', y=1, fontdict={'fontsize': 20})


def plot_pca_var_explained(pca_transformer, figsize=(12, 6)):
    var_ratio = pca_transformer.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_ratio)
    plt.figure(figsize=figsize)
    plt.bar(range(1, len(cum_var_exp) + 1), var_ratio, align='center',
            color='red', label='Individual explained variance')
    plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp,
             where='mid', label='Cumulative explained variance')
    plt.xticks(range(1, len(cum_var_exp)))
    plt.legend(loc='best')
    plt.xlabel('Principal component index', {'fontsize': 14})
    plt.ylabel('Explained variance ratio', {'fontsize': 14})
    plt.title('PCA on training data', {'fontsize': 18})


def plot_feature_imp_v1(
    rf, X_train, y_train, feature_names, mode='sklearn',
    figsize=(12, 6), title='Feature Importance'):
    
    if mode == 'permutation':
        if rf.__class__.__name__ == 'RandomForestRegressor':
            feature_imp = permutation_importances(rf, X_train, y_train, oob_r2)
        else:
            feature_imp = permutation_importances(
                rf, X_train, y_train, oob_accuracy)

    elif mode == 'drop_column':
        feature_imp = drop_column_importances(rf, X_train, y_train)

    elif mode == 'sklearn':
        rf.fit(X_train, y_train)
        feature_imp = rf.feature_importances_

    else:
        raise Exception(
            'mode valid values: permutation, drop_column, sklearn_importance')

    indices = np.argsort(feature_imp)
    names = np.array([feature_names[i] for i in indices])
    plt.figure(figsize=figsize)
    plt.barh(range(len(feature_imp)), feature_imp[indices])
    plt.yticks(range(len(feature_imp)), names, fontsize=16)
    plt.title(title, {'fontsize': 20})


def plot_feature_imp(clf, feature_names, figsize=(12, 6)):
    feature_imp = clf.feature_importances_
    indices = np.argsort(feature_imp)[::-1]
    names = [feature_names[i] for i in indices]
    plt.figure(figsize=figsize)
    plt.bar(range(len(feature_imp)), feature_imp[indices])
    plt.xticks(range(len(feature_imp)), names, rotation=90, fontsize=16)
    plt.title('Feature Importance', {'fontsize': 20})


def plot_validation_curve(
    train_scores, test_scores, param_range, x_label, y_lim=(0, 1),
    title='title', figsize=(12, 8)):
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=figsize)
    plt.plot(param_range, train_mean, label='Training score', color='blue',
             marker='o', markersize=10)
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, label='Testing score', color='red',
             marker='s', markersize=10)
    plt.fill_between(param_range, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='red')
    plt.ylim(y_lim)
    plt.legend()
    plt.xlabel(x_label, {'fontsize': 16})
    plt.title(title, {'fontsize': 20})


def plot_cv_error(cv_error, metric_name='Accuracy', figsize=(12, 8)):
    k = len(cv_error)
    avg_cv_error = cv_error.mean()
    std_cv_error = cv_error.std()
    plt.figure(figsize=figsize)
    plt.plot(range(1, k + 1), cv_error, label='CV error')
    plt.plot([1, k], [avg_cv_error, avg_cv_error],
             'r--', label='Average CV error')
    plt.fill_between(range(1, k + 1), cv_error +
                     std_cv_error, cv_error - std_cv_error, alpha=0.3)
    plt.title(f'{k}-folds CV {metric_name}', fontsize='20')
    plt.legend()
    plt.xticks(range(1, k + 1), range(1, k + 1))


def compute_stable_oob_score(rf, X_train, y_train, trials=5, metric_name='accuracy'):
    oob_score = []
    for i in range(trials):
        rf.fit(X_train, y_train)
        oob_score.append(rf.oob_score_)
    avg_oob_score = np.mean(oob_score)
    std_oob_score = np.std(oob_score)
    print(f'average OOB {metric_name} : {avg_oob_score:.4f} +/- {std_oob_score:.4f}')
