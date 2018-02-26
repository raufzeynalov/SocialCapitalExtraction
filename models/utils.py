import time
from contextlib import contextmanager

import numpy as np
import psycopg2
import psycopg2.extras
from IPython.core.magics.execution import _format_time as format_delta
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from .url_and_media_text_preprocessing import UrlAndMediaTextBooleanExtractor, UrlAndMediaTextStripper
from .text_preprocessing import LengthInWordsTransformer
from .subjectivity_modeller import SubjectivityClassifier


@contextmanager
def timing(prefix=''):
    t0 = time.time()
    yield
    print(' '.join([prefix, 'elapsed time: %s' % format_delta(time.time() - t0)]))


n_topics = 20
#helper function to get all facebook features
def get_features_facebook_qualitative():
    
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

#helper function to get all facebook features
def get_features_facebook_quantitative():
    
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
        (['user_posts_num'], None, {'alias': 'user_posts_num'}),
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


#helper function to get all twitter features
def get_features_twitter_qualitative():
    user_features = CustomDataFrameMapper([
        #(['FOLLOWER_INDEGREE'], None, {'alias': 'followers_num'}),
        (['FOLLOWER_OUTDEGREE'], None, {'alias': 'followees_num'}),
        (['NUM_DAYS_SINCE_SIGNUP'], None, {'alias': 'num_days_since_signup'}),
        #(['RETWEET_INDEGREE_UNWEIGHTED'], None, {'alias': 'retweet_user_num'}),
        (['RETWEET_OUTDEGREE_UNWEIGHTED'], None, {'alias': 'retweeted_by_user_num'}),
        (['REPLY_INDEGREE_UNWEIGHTED'], None, {'alias': 'reply_user_num'}),
        (['REPLY_OUTDEGREE_UNWEIGHTED'], None, {'alias': 'replied_by_user_num'})
    ])
   
    tweet_features = CustomDataFrameMapper([
        #(['NUM_TWEETS'], None, {'alias': 'tweets_num'}),
        (['AVG_NUM_TWEETS_PER_DAY'], None, {'alias': 'avg_num_tweets_per_day'}),
        (['AVG_LENGTH_TWEETS'], None, {'alias': 'avg_length_tweets'}),
        (['NUM_MENTIONS'], None, {'alias': 'num_mentions_in_tweets'}),
        (['NUM_TWEETS_WITH_MENTIONS'], None, {'alias': 'num_tweets_with_mention'}),
        (['NUM_HASHTAGS'], None, {'alias': 'num_hashtags_in_tweets'}),
        (['NUM_TWEETS_WITH_HASHTAGS'], None, {'alias': 'num_tweets_with_hashtag'}),
    ])
                                              
    retweet_features = CustomDataFrameMapper([
        #(['RETWEET_INDEGREE_WEIGHTED'], None, {'alias': 'num_received_retweets'}),
        (['RETWEET_OUTDEGREE_WEIGHTED'], None, {'alias': 'num_posted_retweets'})
    ])
    
    reply_features = CustomDataFrameMapper([
        (['REPLY_INDEGREE_WEIGHTED'], None, {'alias': 'num_received_replies'}),
        (['REPLY_OUTDEGREE_WEIGHTED'], None, {'alias': 'num_sent_replies'})
    ])
    return reply_features, retweet_features, tweet_features, user_features


#helper function to get all twitter features
def get_features_twitter_quantitative():
    user_features = CustomDataFrameMapper([
        (['FOLLOWER_INDEGREE'], None, {'alias': 'followers_num'}),
        (['FOLLOWER_OUTDEGREE'], None, {'alias': 'followees_num'}),
        (['NUM_DAYS_SINCE_SIGNUP'], None, {'alias': 'num_days_since_signup'}),
        #(['RETWEET_INDEGREE_UNWEIGHTED'], None, {'alias': 'retweet_user_num'}),
        (['RETWEET_OUTDEGREE_UNWEIGHTED'], None, {'alias': 'retweeted_by_user_num'}),
        (['REPLY_INDEGREE_UNWEIGHTED'], None, {'alias': 'reply_user_num'}),
        (['REPLY_OUTDEGREE_UNWEIGHTED'], None, {'alias': 'replied_by_user_num'})
    ])
   
    tweet_features = CustomDataFrameMapper([
        (['NUM_TWEETS'], None, {'alias': 'tweets_num'}),
        (['AVG_NUM_TWEETS_PER_DAY'], None, {'alias': 'avg_num_tweets_per_day'}),
        (['AVG_LENGTH_TWEETS'], None, {'alias': 'avg_length_tweets'}),
        (['NUM_MENTIONS'], None, {'alias': 'num_mentions_in_tweets'}),
        (['NUM_TWEETS_WITH_MENTIONS'], None, {'alias': 'num_tweets_with_mention'}),
        (['NUM_HASHTAGS'], None, {'alias': 'num_hashtags_in_tweets'}),
        (['NUM_TWEETS_WITH_HASHTAGS'], None, {'alias': 'num_tweets_with_hashtag'}),
    ])
                                              
    retweet_features = CustomDataFrameMapper([
        #(['RETWEET_INDEGREE_WEIGHTED'], None, {'alias': 'num_received_retweets'}),
        (['RETWEET_OUTDEGREE_WEIGHTED'], None, {'alias': 'num_posted_retweets'})
    ])
    
    reply_features = CustomDataFrameMapper([
        (['REPLY_INDEGREE_WEIGHTED'], None, {'alias': 'num_received_replies'}),
        (['REPLY_OUTDEGREE_WEIGHTED'], None, {'alias': 'num_sent_replies'})
    ])
    return reply_features, retweet_features, tweet_features, user_features



#helper function to get all scientometrics features
def get_features_scientometrics_qualitative():
    
    user_features = CustomDataFrameMapper([
        (['CIT_OUTDEGREE_UNWEIGHTED'], None, {'alias': 'num_author_referenced_by'}),
        (['CIT_OUTDEGREE_WEIGHTED'], None, {'alias': 'num_references'}),
    ])
    
    publication_features = CustomDataFrameMapper([
        #(['PAPER_COUNT'], None, {'alias': 'num_publication'}),
        (['NUM_FIRST_POS'], None, {'alias': 'num_first_pos'}),
        (['NUM_SECOND_POS'], None, {'alias': 'num_second_pos'}),
        #(['NUM_THIRD_POS'], None, {'alias': 'num_third_pos'}),
        #(['NUM_HIGHER_POS'], None, {'alias': 'num_higher_pos'}),
        (['NUM_YEARS_SINCE_FIRST_PUBLICATION'], None, {'alias': 'num_years_since_first_publication'}),
        (['NUM_YEARS_BETWEEN_FIRST_AND_LAST_PUBLICATION'], None, {'alias': 'years_between_first_and_last_pub'}),
        #(['AVG_NUM_PUBLICATIONS_PER_YEAR'], None, {'alias': 'avg_num_publication_per_year'}),
        (['AVG_TITLE_LENGTH'], None, {'alias': 'avg_title_length'}),
        (['AVG_ABSTRACT_LENGTH'], None, {'alias': 'avg_abstract_length'})
    ])
   
    institution_features = CustomDataFrameMapper([
        (['NUM_INSTITUTIONS'], None, {'alias': 'num_institution'}),
        (['NUM_TOP500_INSTITUTIONS'], None, {'alias': 'num_top500_institution'}),
        (['SHANGHAI_RANK'], None, {'alias': 'shanghai_rank'}),
        (['NTU_RANK'], None, {'alias': 'ntu_rank'}),
        (['THE_RANK'], None, {'alias': 'the_rank'}),
        (['SHANGHAI_SCORE'], None, {'alias': 'shanghai_score'}),
        (['NTU_SCORE'], None, {'alias': 'ntu_score'}),
        (['THE_SCORE'], None, {'alias': 'the_score'}),
    ])
                                              
    topic_distribution_features = CustomDataFrameMapper([
        (['TOP_SIM_CORPUS'], None, {'alias': 'topic_sim_corpus'}),
        (['TOP_SIM_UNIFORM'], None, {'alias': 'topic_sim_uniform'}),
        (['TOP SIM PAPERS'], None, {'alias': 'topic_sim_papers'}),
        (['NUM_TOPICS_GREATER_CORPUS'], None, {'alias': 'num_topics_greater_corpus'}),
        (['NUM_TOPICS_GREATER_UNIFORM'], None, {'alias': 'num_topics_greater_uniform'}),
    ])
    
    return topic_distribution_features, institution_features, publication_features, user_features

#helper function to get all scientometrics features
def get_features_scientometrics_quantitative():
    user_features = CustomDataFrameMapper([
        (['CIT_OUTDEGREE_UNWEIGHTED'], None, {'alias': 'num_author_referenced_by'}),
        (['CIT_OUTDEGREE_WEIGHTED'], None, {'alias': 'num_references'}),
    ])
    publication_features = CustomDataFrameMapper([
        (['PAPER_COUNT'], None, {'alias': 'num_publication'}),
        (['NUM_FIRST_POS'], None, {'alias': 'num_first_pos'}),
        (['NUM_SECOND_POS'], None, {'alias': 'num_second_pos'}),
        (['NUM_THIRD_POS'], None, {'alias': 'num_third_pos'}),
        (['NUM_HIGHER_POS'], None, {'alias': 'num_higher_pos'}),
        (['NUM_YEARS_SINCE_FIRST_PUBLICATION'], None, {'alias': 'num_years_since_first_publication'}),
        (['NUM_YEARS_BETWEEN_FIRST_AND_LAST_PUBLICATION'], None, {'alias': 'years_between_first_and_last_pub'}),
        (['AVG_NUM_PUBLICATIONS_PER_YEAR'], None, {'alias': 'avg_num_publication_per_year'}),
        (['AVG_TITLE_LENGTH'], None, {'alias': 'avg_title_length'}),
        (['AVG_ABSTRACT_LENGTH'], None, {'alias': 'avg_abstract_length'})
    ])
   
    institution_features = CustomDataFrameMapper([
        (['NUM_INSTITUTIONS'], None, {'alias': 'num_institution'}),
        (['NUM_TOP500_INSTITUTIONS'], None, {'alias': 'num_top500_institution'}),
        (['SHANGHAI_RANK'], None, {'alias': 'shanghai_rank'}),
        (['NTU_RANK'], None, {'alias': 'ntu_rank'}),
        (['THE_RANK'], None, {'alias': 'the_rank'}),
        (['SHANGHAI_SCORE'], None, {'alias': 'shanghai_score'}),
        (['NTU_SCORE'], None, {'alias': 'ntu_score'}),
        (['THE_SCORE'], None, {'alias': 'the_score'}),
    ])
                                              
    topic_distribution_features = CustomDataFrameMapper([
        (['TOP_SIM_CORPUS'], None, {'alias': 'topic_sim_corpus'}),
        (['TOP_SIM_UNIFORM'], None, {'alias': 'topic_sim_uniform'}),
        (['TOP SIM PAPERS'], None, {'alias': 'topic_sim_papers'}),
        (['TOP_SIM_UNIFORM'], None, {'alias': 'topic_sim_uniform'}),
        (['NUM_TOPICS_GREATER_CORPUS'], None, {'alias': 'num_topics_greater_corpus'}),
        (['NUM_TOPICS_GREATER_UNIFORM'], None, {'alias': 'num_topics_greater_uniform'}),
    ])
    
    return topic_distribution_features, institution_features, publication_features, user_features


def get_svr_features():
    user_features = CustomDataFrameMapper([
        (['user_friends_num'], [StandardScaler()], {'alias': 'friends_num'}),
        (['highest_education_level'], [StandardScaler()]),
        (['education_is_present'], None),
        (['languages_num'], [StandardScaler()]),
        (['language_info_is_present'], None),
    ])
    mention_features = CustomDataFrameMapper([
        (['unique_mention_authors_per_friend'], [StandardScaler()]),
        (['mentions_per_friend'], [StandardScaler()]),
    ])
    post_features = CustomDataFrameMapper([
        (['user_friends_per_post'], [StandardScaler()], {'alias': 'friends_per_post'}),
        (['user_media_to_all_normal_ratio'], [StandardScaler()], {'alias': 'media_to_all_normal_ratio'}),
        (['user_normal_posts_num'], [StandardScaler()], {'alias': 'normal_posts_num'}),
        (['user_life_events_num'], [StandardScaler()], {'alias': 'life_events_num'}),
        (['user_small_posts_num'], [StandardScaler()], {'alias': 'small_posts_num'}),
        (['avg_normal_post_length'], [StandardScaler()], {'alias': 'normal_post_avg_length'}),
    ])
    comment_features = CustomDataFrameMapper([
        (['user_comments_num'], [StandardScaler()], {'alias': 'comments_num'}),
        (['avg_comment_length'], [StandardScaler()], {'alias': 'comment_avg_length'}),
        (['user_likes_per_comment'], [StandardScaler()], {'alias': 'likes_per_user_comment'}),
        (['comments_on_own_posts_num'], [StandardScaler()]),
        (['comments_on_own_life_events_num'], [StandardScaler()]),
    ])

    return comment_features, mention_features, post_features, user_features

def _tree_features_transformations():
   
    user_features_transformations = [
                                        (['user_answers_num'], None),
                                        (['user_questions_num'], None),
                                        (['user_edits_num'], None),
                                        (['user_topics_num'], None),
                                        (['user_followers_num'], None),
                                        (['user_followings_num'], None),
                                        (['user_ff_ratio'], None),
                                        (['user_z_score'], None),
                                        (['user_top_score_ratio'], None),
                                    ] + [
                                        (['user_topic_presense_%d' % i], None) for i in range(1, n_topics + 1)
                                    ] + [
                                        (['user_%s_presense' % tag], None) for tag in ['obj', 'subj']
                                    ]
    question_features_transformations = [
                                            (['question_fetched_answers_num'], None),
                                            (['question_subjectivity'], None),
                                        ] + [
                                            (['lda_%d' % i], None) for i in range(1, n_topics + 1)
                                        ]
    # noinspection PyTypeChecker
    answer_features_transformations = [
        ('answer_content', [UrlAndMediaTextStripper(), LengthInWordsTransformer()], {'alias': 'answer_content_length'}),
        ('answer_content', UrlAndMediaTextBooleanExtractor()),
        ('answer_smog_index', None)
    ]
    time_features_transformations = [
        (['days_rescaled'], None)
    ]
    return answer_features_transformations, question_features_transformations, time_features_transformations, user_features_transformations


def get_general_features_quora(): 
     
    user_features_transformations = [
                                        (['user_answers_num'], None),
                                        (['user_questions_num'], None),
                                        (['user_edits_num'], None),
                                        (['user_topics_num'], None),
                                        (['user_posts_num'], None),
                                        (['user_followers_num'], None),
                                        (['user_followings_num'], None),
                                        (['user_ff_ratio'], None),
                                        (['user_z_score'], None),
                                        (['user_top_score_ratio'], None),
                                    ]
    question_features_transformations = [
                                            (['question_fetched_answers_num'], None), 
                                            (['question_followers_num'], None),
                                            (['question_comments_num'], None)
        
                                        ] 
    # noinspection PyTypeChecker
    answer_features_transformations = [
        ('answer_content', UrlAndMediaTextBooleanExtractor()),
        ('answer_comments_num', None),
        ('answer_smog_index', None),
        ('answer_rank_ratio',None)
    ]
    return answer_features_transformations, question_features_transformations, user_features_transformations

def _svr_features_transformations():
    log_transformer = FunctionTransformer(func=np.log1p)
    user_features_transformations = [
        (['user_answers_num'], [log_transformer, StandardScaler()]),
        (['user_questions_num'], [log_transformer, StandardScaler()]),
        (['user_edits_num'], [log_transformer, StandardScaler()]),
        (['user_topics_num'], [log_transformer, StandardScaler()]),
        (['user_followers_num'], [log_transformer, StandardScaler()]),
        (['user_followings_num'], [log_transformer, StandardScaler()]),
        (['user_ff_ratio'], [log_transformer, StandardScaler()]),
        (['user_z_score'], StandardScaler()),
        (['user_top_score_ratio'], StandardScaler()),
        ] + [
            (['user_topic_presense_%d' % i], StandardScaler()) for i in range(1, n_topics + 1)
        ] + [
            (['user_%s_presense' % tag], StandardScaler()) for tag in ['obj', 'subj']
    ]
    question_features_transformations = [
        (['question_fetched_answers_num'], [log_transformer, StandardScaler()]),
        ('question_title', [SubjectivityClassifier('./data/subjectivity/subj.clf'),
                            StandardScaler()], {'alias': 'question_subjectivity'})
        ] + [
            (['lda_%d' % i], StandardScaler()) for i in range(1, n_topics + 1)
    ]
    answer_features_transformations = [
        ('answer_content', [UrlAndMediaTextStripper(), LengthInWordsTransformer(), log_transformer, StandardScaler()], {'alias': 'answer_content_length'}),
        ('answer_content', UrlAndMediaTextBooleanExtractor()),
        (['answer_smog_index'], StandardScaler())
    ]
    time_features_transformations = [
        (['days_rescaled'], StandardScaler())
    ]
    return answer_features_transformations, question_features_transformations, \
           time_features_transformations, user_features_transformations


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
