import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import re
from IPython.core.magics.execution import _format_time as format_delta


@contextmanager
def timing(prefix=''):
    t0 = time.time()
    yield
    print(' '.join([prefix, 'elapsed time: %s' % format_delta(time.time() - t0)]))


class Preprocessor:
    def __init__(self, posts_data, own_comments_data, comments_data, education_data, languages_data, work_data, mentions_data,
                 log_score=True, rescale_score=True):
        self.posts_data = posts_data.copy()
        self.own_comments_data = own_comments_data.copy()
        self.comments_data = comments_data.copy()
        self.education_data = education_data.copy()
        self.languages_data = languages_data.copy()
        self.work_data = work_data.copy()
        self.mentions_data = mentions_data.copy()
        self.data = None

        self._log_score = log_score
        self._rescale_score = rescale_score

        self._token_pattern = re.compile(r'(?u)\b\w+\b', re.MULTILINE)

    def fit_transform(self):
        print('Preprocessing data...')
        print('Data size: %d posts, %d comments, %d users in posts, %d and %d users in comments' %
              (len(self.posts_data), len(self.own_comments_data),
               len(self.posts_data.groupby(['user_id']).count()),
               len(self.comments_data.groupby(['post_user_id']).count()),
               len(self.own_comments_data.groupby(['comment_user_id']).count())))

        self.compute_score()
        self.log_score()
        self.rescale_score()

        print('\tcalculating post features')
        self.calculate_likes_per_post()
        self.calculate_friends_per_post()
        self.calculate_post_nums_by_type()
        self.calculate_average_length_of_posts()

        print('\tcalculating comment features')
        self.calculate_comments_counters()
        self.calculate_avg_comment_length()
        self.calculate_likes_per_comment()

        print('\tcalculating user features')
        self.calculate_highest_education()
        self.calculate_number_of_languages()
        self.calculate_mentions()

        print('\tmerging features')
        posts_stripped_data = self.posts_data.groupby('user_id').first().reset_index()[['user_id', 'score', 'user_friends_num',
                                                                                        'user_posts_num', #new feature
                                                                                        'user_likes_per_post', 'user_friends_per_post',
                                                                                        'user_media_to_all_normal_ratio',
                                                                                        'user_normal_posts_num', 'user_life_events_num', 'user_small_posts_num',
                                                                                        'avg_normal_post_length']]
        posts_stripped_data.set_index('user_id', inplace=True)
        own_comments_stripped_data = self.own_comments_data.groupby('comment_user_id').first().reset_index()[['comment_user_id', 'user_comments_num',
                                                                                                              'avg_comment_length', 'user_likes_per_comment']]
        own_comments_stripped_data.set_index('comment_user_id', inplace=True)

        comments_stripped_data = self.comments_data.groupby('post_user_id').first().reset_index()[['post_user_id', 'comments_on_own_posts_num',
                                                                                                   'comments_on_own_life_events_num']]
        comments_stripped_data.set_index('post_user_id', inplace=True)

        education_stripped_data = self.education_data.groupby('user_id').first().reset_index()[['user_id', 'highest_education_level',
                                                                                                'education_is_present']]
        education_stripped_data.set_index('user_id', inplace=True)

        languages_stripped_data = self.languages_data.groupby('user_id').first().reset_index()[['user_id', 'languages_num', 'language_info_is_present']]
        languages_stripped_data.set_index('user_id', inplace=True)

        mentions_stripped_data = self.mentions_data.groupby('index').first().reset_index()[['index', 'unique_mention_authors',
                                                                                            'mentions_num', 'own_mentions_num']]
        mentions_stripped_data.set_index('index', inplace=True)

        self.data = posts_stripped_data.merge(own_comments_stripped_data, left_index=True, right_index=True, how='outer')
        self.data = self.data.merge(comments_stripped_data, left_index=True, right_index=True, how='outer')
        self.data = self.data.merge(education_stripped_data, left_index=True, right_index=True)
        self.data = self.data.merge(languages_stripped_data, left_index=True, right_index=True)
        self.data = self.data.merge(mentions_stripped_data, left_index=True, right_index=True, how='outer')

        self.data['unique_mention_authors_per_friend'] = self.data['unique_mention_authors'] / self.data['user_friends_num']
        self.data['mentions_per_friend'] = self.data['mentions_num'] / self.data['user_friends_num']
        self.data['own_mentions_per_friend'] = self.data['own_mentions_num'] / self.data['user_friends_num']

        for column in self.data.columns:
            self.data[column].fillna(0, inplace=True)

        print('\tfinal size: %d' % len(self.data))
        print('...done\n')
        return self.data

    def compute_score(self):
        posts_stats = self.posts_data.groupby(['user_id'], as_index=False)[['post_id']].count()
        posts_stats.rename(index=str, inplace=True, columns={'post_id': 'user_posts_num'})
        self.posts_data = self.posts_data.join(posts_stats.set_index('user_id'), on='user_id')

        likes_stats = self.posts_data.groupby(['user_id'], as_index=False)[['post_likes_num']].sum()
        likes_stats.rename(index=str, inplace=True, columns={'post_likes_num': 'user_likes_received_sum'})
        self.posts_data = self.posts_data.join(likes_stats.set_index('user_id'), on='user_id')
        self.posts_data['score'] = self.posts_data['user_likes_received_sum'].copy()
        #((self.posts_data['user_likes_received_sum'] + 1) / self.posts_data['user_posts_num']) / \
                                   self.posts_data['user_friends_num']

    def rescale_score(self):
        if self._rescale_score:
            min_score = min(self.posts_data['score'])
            max_score = max(self.posts_data['score'])
            print('\trescaling score from [%.2f, %.2f] to [0., 1.]' % (min_score, max_score))
            self.posts_data['score'] = (self.posts_data['score'] - min_score) / (max_score - min_score)

    def log_score(self):
        if self._log_score:
            min_score = min(self.posts_data['score'])
            max_score = max(self.posts_data['score'])
            print('\tapplying log transform on the score; original range is \n\t\t[%.16f, %.16f]' % (min_score, max_score))
            self.posts_data['score'] = np.log1p(self.posts_data['score'] * 10000)
            print('\tnew range is \n\t\t[%.16f, %.16f]' % (min(self.posts_data['score']), max(self.posts_data['score'])))

    def calculate_likes_per_post(self):
        self.posts_data['user_likes_per_post'] = self.posts_data['user_likes_received_sum'] / self.posts_data['user_posts_num']

    def calculate_friends_per_post(self):
        self.posts_data['user_friends_per_post'] = self.posts_data['user_friends_num'] / self.posts_data['user_posts_num']

    def calculate_post_nums_by_type(self):
        # normal: embedded + non-embedded
        post_stats = self.posts_data[(self.posts_data['post_type'] == 2) | (self.posts_data['post_type'] == 5)] \
            .groupby(['user_id'], as_index=False)[['post_id']].count()
        post_stats.rename(index=str, inplace=True, columns={'post_id': 'user_normal_posts_num'})
        self.posts_data = self.posts_data.join(post_stats.set_index('user_id'), on='user_id')

        # count media posts ratio
        post_stats = self.posts_data[(self.posts_data['post_type'] == 5) |
                                     (self.posts_data['post_type'] == 2 &
                                      (self.posts_data['video_id'].isnull() |
                                       self.posts_data['photo_id'].isnull() |
                                       self.posts_data['photo_album_id'].isnull()))].groupby(['user_id'], as_index=False)[['post_id']].count()
        post_stats.rename(index=str, inplace=True, columns={'post_id': 'user_media_posts_num'})
        self.posts_data = self.posts_data.join(post_stats.set_index('user_id'), on='user_id')
        self.posts_data['user_media_to_all_normal_ratio'] = self.posts_data['user_media_posts_num'] / self.posts_data['user_normal_posts_num']
        self.posts_data.loc[self.posts_data['user_normal_posts_num'] == 0, 'user_media_posts_num'] = 0.0

        # life events
        post_stats = self.posts_data[self.posts_data['post_type'] == 4].groupby(['user_id'], as_index=False)[['post_id']].count()
        post_stats.rename(index=str, inplace=True, columns={'post_id': 'user_life_events_num'})
        self.posts_data = self.posts_data.join(post_stats.set_index('user_id'), on='user_id')

        # small posts
        post_stats = self.posts_data[self.posts_data['post_type'] == 3].groupby(['user_id'], as_index=False)[['post_id']].count()
        post_stats.rename(index=str, inplace=True, columns={'post_id': 'user_small_posts_num'})
        self.posts_data = self.posts_data.join(post_stats.set_index('user_id'), on='user_id')

    def calculate_average_length_of_posts(self):
        self.posts_data['post_length'] = self.posts_data['post_text'].apply(lambda text: len(self._token_pattern.findall(text)))

        post_stats = self.posts_data[(self.posts_data['post_type'] == 2) | (self.posts_data['post_type'] == 5)] \
            .groupby(['user_id'], as_index=False)[['post_length']].mean()
        post_stats.rename(index=str, inplace=True, columns={'post_length': 'avg_normal_post_length'})
        self.posts_data = self.posts_data.join(post_stats.set_index('user_id'), on='user_id')

    def calculate_comments_counters(self):
        comments_stats = self.comments_data[(self.comments_data['post_type'] == 2) | (self.comments_data['post_type'] == 5)] \
            .groupby(['post_user_id'], as_index=False)[['comment_id']].count()
        comments_stats.rename(index=str, inplace=True, columns={'comment_id': 'comments_on_own_posts_num'})
        self.comments_data = self.comments_data.join(comments_stats.set_index('post_user_id'), on='post_user_id')

        comments_stats = self.comments_data[self.comments_data['post_type'] == 4] \
            .groupby(['post_user_id'], as_index=False)[['comment_id']].count()
        comments_stats.rename(index=str, inplace=True, columns={'comment_id': 'comments_on_own_life_events_num'})
        self.comments_data = self.comments_data.join(comments_stats.set_index('post_user_id'), on='post_user_id')

        comments_stats = self.own_comments_data.groupby(['comment_user_id'], as_index=False)[['comment_id']].count()
        comments_stats.rename(index=str, inplace=True, columns={'comment_id': 'user_comments_num'})
        self.own_comments_data = self.own_comments_data.join(comments_stats.set_index('comment_user_id'), on='comment_user_id')

    def calculate_avg_comment_length(self):
        self.own_comments_data['comment_length'] = self.own_comments_data['comment_text'].apply(lambda text: len(self._token_pattern.findall(text)))

        comments_stats = self.own_comments_data.groupby(['comment_user_id'], as_index=False)[['comment_length']].mean()
        comments_stats.rename(index=str, inplace=True, columns={'comment_length': 'avg_comment_length'})
        self.own_comments_data = self.own_comments_data.join(comments_stats.set_index('comment_user_id'), on='comment_user_id')

    def calculate_likes_per_comment(self):
        likes_stats = self.own_comments_data.groupby(['comment_user_id'], as_index=False)[['comment_likes_num']].sum()
        likes_stats.rename(index=str, inplace=True, columns={'comment_likes_num': 'user_comment_likes_received_sum'})
        self.own_comments_data = self.own_comments_data.join(likes_stats.set_index('comment_user_id'), on='comment_user_id')
        self.own_comments_data['user_likes_per_comment'] = self.own_comments_data['user_comment_likes_received_sum'] / self.own_comments_data['user_comments_num']

    def calculate_highest_education(self):
        self.education_data.loc[self.education_data['education_type'] == 'Schule', 'education_type'] = 'School'
        self.education_data.loc[self.education_data['education_type'] == 'Hochschule', 'education_type'] = 'College'

        self.education_data.loc[self.education_data['education_type'].str.contains('Bachelor', case=False, na=False), 'education_type'] = 'College'
        self.education_data.loc[self.education_data['education_type'].str.contains('BA', case=True, na=False), 'education_type'] = 'College'
        self.education_data.loc[self.education_data['education_type'].str.contains('B.', case=True, na=False), 'education_type'] = 'College'

        self.education_data.loc[self.education_data['education_type'].str.contains('master', case=False, na=False), 'education_type'] = 'Graduate School'
        self.education_data.loc[self.education_data['education_type'].str.contains('M.', case=True, na=False), 'education_type'] = 'Graduate School'

        self.education_data.loc[self.education_data['education_type'].str.contains('D.', case=True, na=False), 'education_type'] = 'PhD'
        self.education_data.loc[self.education_data['education_type'].str.contains('Doctor', case=False, na=False), 'education_type'] = 'PhD'
        self.education_data.loc[self.education_data['education_type'].str.contains('PhD', case=False, na=False), 'education_type'] = 'PhD'
        self.education_data.loc[self.education_data['education_type'].str.contains('Ph.D.', case=False, na=False), 'education_type'] = 'PhD'
        self.education_data.loc[self.education_data['education_type'].str.contains('Dr.', case=False, na=False), 'education_type'] = 'PhD'

        self.education_data.loc[~self.education_data['education_type'].isin(['PhD', 'Graduate School', 'College', 'School']) &
                                self.education_data['education_name'].str.contains('High', case=False, na=False), 'education_type'] = 'School'
        self.education_data.loc[~self.education_data['education_type'].isin(['PhD', 'Graduate School', 'College', 'School']) &
                                self.education_data['education_name'].str.contains('College', case=False, na=False), 'education_type'] = 'College'
        self.education_data.loc[~self.education_data['education_type'].isin(['PhD', 'Graduate School', 'College', 'School']) &
                                self.education_data['education_name'].str.contains('Universit', case=False, na=False), 'education_type'] = 'College'
        self.education_data.loc[~self.education_data['education_type'].isin(['PhD', 'Graduate School', 'College', 'School']) &
                                self.education_data['education_name'].str.contains('Institute', case=False, na=False), 'education_type'] = 'College'

        self.education_data.loc[~self.education_data['education_type'].isin(['PhD', 'Graduate School', 'College', 'School']) &
                                self.education_data['education_name'].isnull(), 'education_type'] = 'Unknown'
        self.education_data.loc[~self.education_data['education_type'].isin(['PhD', 'Graduate School', 'College', 'School']) &
                                self.education_data['education_name'], 'education_type'] = 'Unknown'

        self.education_data.loc[self.education_data['education_type'] == 'Unknown', 'education_level'] = 0
        self.education_data.loc[self.education_data['education_type'] == 'School', 'education_level'] = 1
        self.education_data.loc[self.education_data['education_type'] == 'College', 'education_level'] = 2
        self.education_data.loc[self.education_data['education_type'] == 'Graduate School', 'education_level'] = 3
        self.education_data.loc[self.education_data['education_type'] == 'PhD', 'education_level'] = 4

        ed_stats = self.education_data.groupby(['user_id'], as_index=False)['education_level'].max()
        ed_stats.rename(index=str, inplace=True, columns={'education_level': 'highest_education_level'})
        self.education_data = self.education_data.join(ed_stats.set_index('user_id'), on='user_id')
        self.education_data.loc[self.education_data['highest_education_level'] == 0, 'education_is_present'] = False
        self.education_data.loc[self.education_data['highest_education_level'] > 0, 'education_is_present'] = True

    def calculate_number_of_languages(self):
        self.languages_data.loc[self.languages_data['spoken_language'].str.contains('Englisch', na=False, case=False), 'spoken_language'] = 'Englisch'
        self.languages_data.drop_duplicates(inplace=True)
        lang_stats = self.languages_data.groupby(['user_id'], as_index=False)['spoken_language'].count()
        lang_stats.rename(index=str, inplace=True, columns={'spoken_language': 'languages_num'})
        self.languages_data = self.languages_data.join(lang_stats.set_index('user_id'), on='user_id')
        self.languages_data.loc[self.languages_data['languages_num'] == 0, 'language_info_is_present'] = False
        self.languages_data.loc[self.languages_data['languages_num'] > 0, 'language_info_is_present'] = True

    def calculate_mentions(self):
        mention_stats = self.mentions_data.groupby(['mentioned_user_id'], as_index=False)['post_id'].count()
        mention_stats.rename(index=str, inplace=True, columns={'post_id': 'mentions_num'})
        self.mentions_data = self.mentions_data.join(mention_stats.set_index('mentioned_user_id'), on='mentioned_user_id')

        mention_stats = self.mentions_data[['mentioned_user_id', 'post_user_id']].drop_duplicates()
        mention_stats = mention_stats.groupby(['mentioned_user_id'], as_index=False)['post_user_id'].count()
        mention_stats.rename(index=str, inplace=True, columns={'post_user_id': 'unique_mention_authors'})
        self.mentions_data = self.mentions_data.join(mention_stats.set_index('mentioned_user_id'), on='mentioned_user_id')

        mention_stats = self.mentions_data.groupby(['post_user_id'], as_index=False)['post_id'].count()
        mention_stats.rename(index=str, inplace=True, columns={'post_id': 'own_mentions_num'})
        mentions_indexed_data = self.mentions_data.set_index('mentioned_user_id')
        self.mentions_data = mentions_indexed_data.merge(mention_stats.set_index('post_user_id'), left_index=True, right_index=True)
        self.mentions_data.loc[self.mentions_data['own_mentions_num'].isnull(), 'own_mentions_num'] = 0
        self.mentions_data.reset_index(level=0, inplace=True)


def avg_datetime(series):
    dt_min = series.min()
    deltas = [(x - dt_min).value for x in series]
    s = 0
    for delta in deltas:
        s += delta
    average_delta = s / len(deltas)
    result = dt_min + pd.Timedelta(average_delta)
    return datetime(result.year, result.month, result.day, 0, 0, 0)
