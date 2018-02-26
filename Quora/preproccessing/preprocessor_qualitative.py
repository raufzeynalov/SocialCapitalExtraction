from datetime import datetime

import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

from readability import SMOGCalculator
from subjectivity_modeller import SubjectivityClassifier
from url_and_media_text_preprocessing import UrlAndMediaTextStripper
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from models.utils import timing


class Preprocessor:
    def __init__(self, data, filter_downvoted=True, log_score=True, rescale_score=True, readability=True,
                 lda_data=None, min_views_num=1, min_question_answers_fetched=1, min_user_answers_fetched=20):

        self.original_data = data.copy()
        self.data = self.original_data

        self._filter_downvoted = filter_downvoted
        self._log_score = log_score
        self._rescale_score = rescale_score
        self._min_views_num = min_views_num
        self._min_question_answers_fetched = min_question_answers_fetched
        self._min_user_answers_fetched = min_user_answers_fetched
        self._lda_data = lda_data
        self._readability = readability

        self.n_topics = None

    def fit_transform(self):
        print('Preprocessing data...')
        self.filter_downvoted()
        self.filter_nans()
        self.filter_by_views()
        self.filter_by_answers_fetched()
        self.filter_by_processed_user_answers_num()
        self.preprocess_timestamps()
        print('Data size: %d' % len(self.data))

        self.compute_score()
        self.log_score()
        self.rescale_score()

        print('\tcalculating features')
        self.calculate_z_score()
        self.calculate_ff_ratio()
        self.calculate_rank_ratio()
        self.calculate_top_rated_ratio()
        self.calculate_question_subjectivity()
        self.calculate_smog_min_age()
        self.calculate_user_subjectivity_presense()

        self.join_with_lda_data()
        self.calculate_user_topic_presense()

        print('...done\n')
        return self.data

    def join_with_lda_data(self):
        if self._lda_data is not None:
            print('\tjoining with question lda information')
            self.n_topics = self._lda_data.shape[1] - 2
            self.data = self.data.join(self._lda_data.set_index('question_id'), on='question_id')
            print('\tremoving rows for which LDA is not available: ', end='')
            prev_len = len(self.data)
            self.data = self.data[self.data['lda_1'].isnull() == False]
            new_len = len(self.data)
            print('%d records removed' % (prev_len - new_len))

    def compute_score(self):
        self.data['score'] = (self.data['answer_upvotes_num'] + 1) / self.data['answer_views_num']
        self.data[self.data['answer_is_downvoted'] == True] = 0.0

    def rescale_score(self):
        if self._rescale_score:
            min_score = min(self.data['score'])
            max_score = max(self.data['score'])
            print('\trescaling score from [%.2f, %.2f] to [0., 1.]' % (min_score, max_score))
            self.data['score'] = (self.data['score'] - min_score) / (max_score - min_score)

    def log_score(self):
        if self._log_score:
            min_score = min(self.data['score'])
            max_score = max(self.data['score'])
            print('\tapplying log transform on the score; original range is \n\t\t[%.16f, %.16f]' % (min_score, max_score))
            self.data['score'] = np.log1p(self.data['score'] * 10000)
            print('\tnew range is \n\t\t[%.16f, %.16f]' % (min(self.data['score']), max(self.data['score'])))

    def calculate_ff_ratio(self):
        self.data['user_ff_ratio'] = (self.data['user_followers_num'] + 1) / (self.data['user_followings_num'] + 1)

    def calculate_z_score(self):
        self.data['user_z_score'] = (self.data['user_answers_num'] - self.data['user_questions_num']) / \
                                    np.sqrt(self.data['user_answers_num'] + self.data['user_questions_num'])

        self.data.loc[self.data['user_answers_num'] + self.data['user_questions_num'] == 0, 'user_z_score'] = 0.0

    def calculate_rank_ratio(self):
        grouped_by_question = self.data.groupby(['question_id'], as_index=False)
        rank_stats = grouped_by_question[['answer_rank']].max()
        rank_stats.rename(index=str, inplace=True, columns={
            'answer_rank': 'max_answer_rank'
        })
        self.data = self.data.join(rank_stats.set_index('question_id'), on='question_id')
        self.data['answer_rank_ratio'] = self.data['answer_rank'] / self.data['max_answer_rank']

    def preprocess_timestamps(self):
        # substitute timestamps of 1970-01-01 with "average" timestamp
        avg_answer_timestamp = avg_datetime(self.data[self.data['answer_timestamp'] > pd.Timestamp('19700101')]['answer_timestamp'])
        self.data.loc[self.data['answer_timestamp'] == pd.Timestamp('19700101'), 'answer_timestamp'] = avg_answer_timestamp

        # convert to number of days since the beginning of era
        self.data['days_since_epoch'] = self.data['answer_timestamp'].apply(lambda x: float(x.value / (1e9 * 24 * 3600)))
        self.data['days_rescaled'] = self.data['days_since_epoch'] - min(self.data['days_since_epoch'])

    def filter_by_processed_user_answers_num(self):
        print('\tremoving answers from users with < %d processed answers' % self._min_user_answers_fetched)

        grouped_by_user = self.data.groupby(['user_id'], as_index=False)
        answer_counts = grouped_by_user[['answer_id']].count()
        answer_counts.rename(index=str, inplace=True, columns={
            'answer_id': 'user_fetched_answers_num'
        })
        self.data = self.data.join(answer_counts.set_index('user_id'), on='user_id')
        self.data = self.data[(self.data['user_fetched_answers_num'] >= self._min_user_answers_fetched) |
                              (self.data['user_fetched_answers_num'] >= np.ceil(self.data['user_answers_num'] * 0.9))]

    def filter_downvoted(self):
        if self._filter_downvoted:
            print('\tremoving downvoted_answers')
            self.data = self.data[self.data['answer_is_downvoted'] == False]

    def filter_by_views(self):
        print('\tremoving answers with < %d views: ' % self._min_views_num, end='')
        prev_len = len(self.data)
        self.data = self.data[self.data['answer_views_num'] >= self._min_views_num]
        new_len = len(self.data)
        print('%d records removed (%.1f%%)' % (prev_len - new_len, 100 * (prev_len - new_len) / float(prev_len)))

    def filter_by_answers_fetched(self):
        print('\tremoving questions with < %d fetched answers' % self._min_question_answers_fetched)
        self.data = self.data[self.data['question_fetched_answers_num'] >= self._min_question_answers_fetched]

    def calculate_top_rated_ratio(self):
        sorted_data = self.data.sort_values('score', ascending=False)
        top_users_id = sorted_data.groupby('question_id', as_index=False).first()[['user_id', 'question_id']]
        top_users_id.rename(index=str, inplace=True, columns={'user_id': 'question_top_user_id'})

        temp = self.data.join(top_users_id.set_index('question_id'), on='question_id')
        grouped_by_user = temp.groupby(['user_id'], as_index=False)
        top_score_ratios = pd.DataFrame({
            'user_id': grouped_by_user.first()['user_id'],
            'user_top_score_ratio': grouped_by_user.apply(
                lambda user_group: len(user_group[user_group['user_id'] == user_group['question_top_user_id']]) / float(len(user_group))
            )
        })
        self.data = self.data.join(top_score_ratios.set_index('user_id'), on='user_id')

    def calculate_user_topic_presense(self):
        if self._lda_data is not None:
            print('\tcalculating user topic presense')
            for i in range(1, self.n_topics + 1):
                grouped_by_user = self.data.groupby(['user_id'], as_index=False)
                user_summed_stats = grouped_by_user[['lda_%d' % i]] \
                    .sum() \
                    .rename(index=str, columns={'lda_%d' % i: 'user_topic_presense_%d' % i})
                self.data = self.data.join(user_summed_stats.set_index('user_id'), on='user_id')

                self.data['user_topic_presense_%d' % i] = self.data[['user_topic_presense_%d' % i, 'user_fetched_answers_num']] \
                    .apply(lambda row: row['user_topic_presense_%d' % i] / row['user_fetched_answers_num'], axis=1)

    def calculate_user_subjectivity_presense(self):
        print('\tcalculating user subjectivity presense')
        objective_data = self.data[self.data['question_subjectivity'] < 0.5]
        subjective_data = self.data[self.data['question_subjectivity'] >= 0.5]
        for tag, data in zip(['obj', 'subj'], [objective_data, subjective_data]):
            grouped_by_user = data.groupby(['user_id'], as_index=False)
            user_count_stats = grouped_by_user[['question_subjectivity']] \
                .count() \
                .rename(index=str, columns={'question_subjectivity': 'user_%s_presense' % tag})
            self.data = self.data.join(user_count_stats.set_index('user_id'), on='user_id')

            self.data['user_%s_presense' % tag] = self.data[['user_%s_presense' % tag, 'user_fetched_answers_num']] \
                .apply(lambda row: row['user_%s_presense' % tag] / row['user_fetched_answers_num'], axis=1)
        # if result is Nan, substitute it with 0
        self.data.loc[np.isnan(self.data['user_obj_presense']), 'user_obj_presense'] = 0.0
        self.data.loc[np.isnan(self.data['user_subj_presense']), 'user_subj_presense'] = 0.0

    def filter_nans(self):
        print('\tremoving questions without fetched title: ', end='')
        prev_len = len(self.data)
        self.data = self.data[self.data['question_title'] != '']
        new_len = len(self.data)
        print('%d records removed' % (prev_len - new_len))

    def calculate_question_subjectivity(self):
        print('\tcalculating questions subjectivity')
        mapper = DataFrameMapper([
            ('question_title', SubjectivityClassifier('../data/subjectivity/subj.clf'), {'alias': 'question_subjectivity'})
        ])
        self.data['question_subjectivity'] = mapper.fit_transform(self.data)

    def calculate_smog_min_age(self):
        if self._readability:
            with timing('\t\tSMOG'):
                mapper = DataFrameMapper([
                    ('answer_id', None),
                    ('answer_content', [UrlAndMediaTextStripper(), SMOGCalculator()], {'alias': 'answer_smog'})
                ], df_out=True)
                result = mapper.fit_transform(self.data)
                self.data = self.data.join(result.set_index('answer_id'), on='answer_id')


def avg_datetime(series):
    dt_min = series.min()
    deltas = [(x - dt_min).value for x in series]
    s = 0
    for delta in deltas:
        s += delta
    average_delta = s / len(deltas)
    result = dt_min + pd.Timedelta(average_delta)
    return datetime(result.year, result.month, result.day, 0, 0, 0)
