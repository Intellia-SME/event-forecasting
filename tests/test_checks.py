from __future__ import annotations

from unittest import TestCase

import pandas as pd

from event_correlation.algorithms import CumsumStepwiseCorrelation


class TestCumsumStepwiseCorrelation(TestCase):
    def test_always_passes(self):
        input = pd.DataFrame(
            {
                "col1": [1, 12, 23, 4, 50, 6, 7, 88, 9, 10, 30],
            }
        )
        prediction = CumsumStepwiseCorrelation().fit(input).predict([[40]])
        self.assertEqual(prediction, [[0]])
