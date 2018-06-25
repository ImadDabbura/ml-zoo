import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
