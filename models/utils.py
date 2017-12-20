import time
from contextlib import contextmanager

import numpy as np
import psycopg2
import psycopg2.extras
from IPython.core.magics.execution import _format_time as format_delta
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn_pandas import DataFrameMapper


@contextmanager
def timing(prefix=''):
    t0 = time.time()
    yield
    print(' '.join([prefix, 'elapsed time: %s' % format_delta(time.time() - t0)]))


class ListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(ListTransformer, self).__init__()

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        result = np.zeros((X.shape[0], len(X[0])))
        for i in range(X.shape[0]):
            result[i] = np.array(X[i])
        return result


class CustomDataFrameMapper(DataFrameMapper):
    def get_feature_names(self):
        return self.transformed_names_


class PsqlReader:
    def __init__(self):
        self.conn = psycopg2.connect("dbname='quora' user='aynroot' host='localhost' port=5432 password='secret'")
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def fetch_all(self, sql_query):
        self.cursor.execute(sql_query)
        return self.cursor.fetchall()

    def fetch_one(self, sql_query):
        self.cursor.execute(sql_query)
        return self.cursor.fetchone()

    def execute(self, sql_query):
        self.cursor.execute(sql_query)
