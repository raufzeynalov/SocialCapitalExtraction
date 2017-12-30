import re

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class LengthInWordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(LengthInWordsTransformer, self).__init__()
        self.token_pattern = re.compile(r'(?u)\b\w+\b', re.MULTILINE)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        result = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            result[i] = np.array([len(self.token_pattern.findall(X[i]))])
        return result
