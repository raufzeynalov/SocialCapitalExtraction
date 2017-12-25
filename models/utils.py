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

    
#helper function to get all facebook features
def get_features():
    user_features = CustomDataFrameMapper([
        (['user_friends_num'], None, {'alias': 'friends_num'}),
        (['highest_education_level'], None),
        (['education_is_present'], None),
        (['languages_num'], None),
        (['language_info_is_present'], None),
    ])
    mention_features = CustomDataFrameMapper([
        (['unique_mention_authors_per_friend'], None),
        (['mentions_per_friend'], None),
    ])
    post_features = CustomDataFrameMapper([
        (['user_friends_per_post'], None, {'alias': 'friends_per_post'}),
        (['user_media_to_all_normal_ratio'], None, {'alias': 'media_to_all_normal_ratio'}),
        (['user_normal_posts_num'], None, {'alias': 'normal_posts_num'}),
        (['user_life_events_num'], None, {'alias': 'life_events_num'}),
        (['user_small_posts_num'], None, {'alias': 'small_posts_num'}),
        (['avg_normal_post_length'], None, {'alias': 'normal_post_avg_length'}),
    ])
    comment_features = CustomDataFrameMapper([
        (['user_comments_num'], None, {'alias': 'comments_num'}),
        (['avg_comment_length'], None, {'alias': 'comment_avg_length'}),
        (['user_likes_per_comment'], None, {'alias': 'likes_per_user_comment'}),
        (['comments_on_own_posts_num'], None),
        (['comments_on_own_life_events_num'], None),
    ])
    return comment_features, mention_features, post_features, user_features

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
