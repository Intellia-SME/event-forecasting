from __future__ import annotations

import pandas as pd

from event_correlation.algorithms import CumsumStepwiseCorrelation

input = pd.DataFrame(
    {
        "col1": [1, 12, 23, 4, 50, 6, 7, 88, 9, 10, 30],
        "col2": [4, 4, 4, 4, 4, 4, 4, 4, 400, 4, 21],
        "col3": [-10, -20, -10, -20, -10, -200, -10, -20, -10, -20, 23],
        "col4": [5, 10, 15, 200, 25, 30, 35, 40, 45, 50, 32],
    }
)

input = input.iloc[:, :4]
predict_input = input[1:5]


prediction = CumsumStepwiseCorrelation().fit(input).predict(predict_input)
print(prediction)
