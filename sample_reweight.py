from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.ensemble import GradientBoostingClassifier
from typing import List, Dict
import torch
import torch.nn.functional as F
import numpy as np
import math


# Not Implemented yet
# class PipelineRFE(Pipeline):
#     # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
#     def fit(self, X, y=None, **fit_params):
#         super(PipelineRFE, self).fit(X, y, **fit_params)
#         self.feature_importances_ = self.steps[-1][-1].feature_importances_
#         return self


class SampleReweight:
    def __init__(self, X: np.array, y: np.array, a1: float, a2: float, b: int, gamma: float):
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.gamma = gamma

        self.X = X
        self.y = torch.tensor(y, dtype=torch.long).squeeze()

        self.C = None

        self.sample_size = X.shape[0]
        self.k = 1

        self.range_to_bin = [(i * (self.sample_size // self.b),
                              (i + 1 + int(i == self.b - 1)) * (self.sample_size // self.b)) for i in range(self.b)]

    def _get_weight(self, l: np.array) -> np.array:
        head = math.ceil(self.C.shape[1] * 0.1)

        l_normed = self.norm(l)
        c_normed = self.norm(self.C)
        # print(self.C.shape)
        # print(c_normed.shape)

        h_1 = - l_normed
        h_2 = self.norm(
            (np.mean(c_normed[:, :head], axis=1) / np.mean(c_normed[:, -head:], axis=1)).reshape(self.sample_size, -1))

        h = self.a1 * h_1 + self.a2 * h_2
        h = h.squeeze()

        sample_to_bin, bins = self._get_bins(h)
        w = np.zeros(self.sample_size)

        for i in range(self.sample_size):
            w[i] = 1 / (self.gamma ** self.k * bins[sample_to_bin[i]] + 0.1)

        return w

    def _get_bins(self, h: np.array) -> (Dict[int, int], List[np.array]):
        ranks = h.argsort()
        bins = [[] for i in range(self.b)]

        idx_to_bin = {}

        for sample_idx in range(ranks.shape[0]):
            for bin_idx, (low, high) in enumerate(self.range_to_bin):
                if low <= ranks[sample_idx] < high:
                    idx_to_bin[sample_idx] = bin_idx
                    bins[bin_idx].append(h[sample_idx])
                    break

        bins = [np.mean(np.asarray(_bin)) for _bin in bins]

        return idx_to_bin, bins

    def cal_sample_loss(self, model) -> np.array:
        if isinstance(model, GradientBoostingClassifier):
            with torch.no_grad():
                predictions = torch.tensor(model.predict_log_proba(self.X))
                # print(predictions.size(), self.y.size())
        else:
            assert isinstance(model, torch.nn.Module)
            with torch.no_grad():
                X_pred = torch.from_numpy(self.X)
                predictions = F.log_softmax(model(X_pred), dim = 1)
                
        loss = self.criterion(predictions, self.y).unsqueeze(1)
        return loss.numpy()

    def reweight(self, model, staged_pred: List[np.array]) -> np.array:
        self.register_C(staged_pred)

        loss = self.cal_sample_loss(model)
        weight = self._get_weight(loss)
        self.k += 1

        return weight

    def register_C(self, staged_pred: List[np.array]):
        self.C = np.zeros((len(staged_pred), self.sample_size))

        with torch.no_grad():
            for i, pred in enumerate(staged_pred):
                self.C[i] = self.criterion(torch.tensor(pred), self.y)

        self.C = self.C.transpose()

    @staticmethod
    def criterion(*args, weight=None, reduction='none'):
        assert len(args) == 2
        return F.nll_loss(*args, weight=weight, reduction=reduction)

    @staticmethod
    def norm(array: np.array) -> np.array:
        ranks = array.argsort(axis=0) + 1
        norm = ranks / array.shape[0]
        return norm
