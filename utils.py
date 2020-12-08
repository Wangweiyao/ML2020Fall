import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from typing import List
from sklearn.ensemble import GradientBoostingClassifier
from torch import nn
from scipy.spatial.distance import cdist


def plot_roc(labels, scores):
    lw = 2
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.3f)' % (roc_auc))
    # plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    # plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")


def eval_model(models: List, X_test: np.ndarray, feature_selected=None, mode='single'):
    """

    :param models: List[CLASSIFIERS]
    :param X_test:
    :param feature_selected: List[List] of selected features for feature selection, else None
    :param mode: 'single' or 'all', 'single' for a single ensemble, 'all' for all sub-ensemble models.
    :return:
    """
    decision_list = []

    if isinstance(models[0], GradientBoostingClassifier):
        for i, model in enumerate(models):
            X = X_test if feature_selected is None else X_test[:, feature_selected[i]]
            decision_list.append(model.decision_function(X))

    elif isinstance(models[0], nn.Module):
        with torch.no_grad():
            for i, model in enumerate(models):
                model.eval()
                X = X_test if feature_selected is None else X_test[:, feature_selected[i]]
                X_test_tensor = torch.from_numpy(X)
                # softmax = torch.nn.functional.softmax(clfs_SR_FS[i](X_test_tensor), dim = 1)
                # decision_list.append(softmax.numpy()[:,1])
                decision_list.append((model(X_test_tensor)).numpy()[:, 1])

    else:
        raise TypeError("Model not recognized")

    print(len(decision_list))
    print(np.asarray(decision_list).shape)
    decision_list = np.asarray(decision_list)
    if mode == 'single':
        y_score_single = np.mean(decision_list, axis=0)
        print(y_score_single)
        return y_score_single
    else:
        y_scores = [np.mean(decision_list[i:, :], axis=0) for i in range(decision_list.shape[0])]
        print(y_scores)
        return y_scores


def get_model_output(models: List, X_test: np.ndarray, feature_selected=None, mode='single'):
    """

    :param models: List[CLASSIFIERS]
    :param X_test:
    :param feature_selected: List[List] of selected features for feature selection, else None
    :param mode: 'single' or 'all', 'single' for a single ensemble, 'all' for all sub-ensemble models.
    :return:
    """
    decision_list = []

    if isinstance(models[0], GradientBoostingClassifier):
        for i, model in enumerate(models):
            X = X_test if feature_selected is None else X_test[:, feature_selected[i]]
            decision_list.append(model.predict_proba(X))

    elif isinstance(models[0], nn.Module):
        with torch.no_grad():
            for i, model in enumerate(models):
                model.eval()
                X = X_test if feature_selected is None else X_test[:, feature_selected[i]]
                X_test_tensor = torch.from_numpy(X)
                # softmax = torch.nn.functional.softmax(clfs_SR_FS[i](X_test_tensor), dim = 1)
                # decision_list.append(softmax.numpy()[:,1])
                decision_list.append((model(X_test_tensor)).numpy())

    else:
        raise TypeError("Model not recognized")

    print(len(decision_list))
    print(np.asarray(decision_list).shape)
    decision_list = np.asarray(decision_list)
    if mode == 'single':
        y_predict = np.mean(decision_list, axis=0)
    else:
        y_predict = [np.argmax(np.mean(decision_list[i:, :], axis=0), axis=1) for i in
                     range(decision_list.shape[0])]
    print(y_predict)
    return y_predict


def filter_by_corr(array, threshold=0.5):
    groups = {}
    grouped = {}
    # print(array.shape)
    for i in range(array.shape[1] - 1):
        if i not in grouped:
            corr_array = 1 - cdist(array[:, i:i + 1].T, array[:, i:].T, metric='correlation')[0]
            high_corr = np.where(corr_array > threshold)[0]
            if len(high_corr) > 0:
                np.random.shuffle(high_corr)
                high_corr = [i + j for j in list(high_corr)]
                # print(high_corr)
                groups[high_corr[0]] = high_corr.copy()
                for elem in high_corr:
                    grouped[elem] = 1
    return list(groups.keys())
