from __future__ import annotations

import random

import numpy as np


class CumsumStepwiseCorrelation:
    def __init__(self):
        self._graph = {}
        pass

    def _updateGraph(self, prevState, event):
        if event not in self._graph:
            self._graph[event] = {}
        if event not in self._graph[prevState]:
            self._graph[prevState][event] = 0
        self._graph[prevState][event] += 1

    def _calculate_cum_sum(self, column):
        P = N = 0
        m = np.abs(column.mean())
        kpos = kneg = m / 2
        thpos = thneg = 2 * m
        s = []
        for num in column:
            spos = sneg = 0
            P = max(0, num - (m - kpos) + P)
            N = min(0, num - (m + kneg) + N)
            if P > thpos:
                spos = 1
                P = N = 0
            if N < -thneg:
                sneg = 1
                P = N = 0
            s.append(spos or sneg)
        return s

    def _predict_from_graph(self, event):
        if event in self._graph:
            if self._graph[event].values():
                maxi = max(self._graph[event].values())
                return random.choice(
                    [k for (k, v) in self._graph[event].items() if v == maxi]
                )
        return "0" * len(event)

    def fit(self, X):
        """Fit the event-correlation classifier from the training dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : graph
            The fitted graph of stepwise-correlation algorithm.
        """
        self._X = X
        cumsum_table = np.apply_along_axis(self._calculate_cum_sum, 0, X)
        events = np.apply_along_axis(
            lambda a: "".join(map(str, a)), 1, cumsum_table.astype(int)
        )
        self._graph[events[0]] = {}
        for (index, _) in enumerate(events[1:]):
            self._updateGraph(events[index], events[index + 1])
        return self

    def predict(self, X):
        """
        Predict the next events for the provided data.
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Next event for each data sample.
        """
        new_array = np.append(self._X, X, axis=0)
        cumsum_table = np.apply_along_axis(self._calculate_cum_sum, 0, new_array)
        predicted_samples = cumsum_table[-len(X) :]
        new_events = np.apply_along_axis(
            lambda a: "".join(map(str, a)), 1, predicted_samples.astype(int)
        )
        results = []
        for event in new_events:
            prediction = self._predict_from_graph(event)
            results.append(np.fromiter(prediction, dtype=int))
        return np.array(results)
