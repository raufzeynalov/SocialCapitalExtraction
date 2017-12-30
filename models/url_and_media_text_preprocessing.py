import re
import numpy as np
import pandas as pd
import sklearn_pandas

from sklearn.base import TransformerMixin, BaseEstimator

URL_PATTERN = r'(^|\s)(%URL%)($|\s)'
MEDIA_PATTERN = r'(^|\s)(%MEDIA%)($|\s)'


class UrlAndMediaTextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(UrlAndMediaTextExtractor, self).__init__()
        self.classes_ = ['urls_num', 'media_num']

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        url_regex = re.compile(URL_PATTERN, re.MULTILINE)
        media_regex = re.compile(MEDIA_PATTERN, re.MULTILINE)

        result = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            urls = url_regex.findall(X[i])
            media = media_regex.findall(X[i])
            result[i] = np.array([
                len(urls) if urls is not None else 0,
                len(media) if media is not None else 0
            ])
        return result


class UrlAndMediaTextBooleanExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(UrlAndMediaTextBooleanExtractor, self).__init__()
        self.classes_ = ['urls_presence', 'media_presence']

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        url_regex = re.compile(URL_PATTERN, re.MULTILINE)
        media_regex = re.compile(MEDIA_PATTERN, re.MULTILINE)

        result = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            result[i] = np.array([
                1 if url_regex.search(X[i]) is not None else 0,
                1 if media_regex.search(X[i]) is not None else 0,
            ])
        return result


class UrlAndMediaTextStripper(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        url_regex = re.compile(URL_PATTERN, re.MULTILINE)
        media_regex = re.compile(MEDIA_PATTERN, re.MULTILINE)

        result = np.empty((X.shape[0],), dtype=X.dtype)
        for i in range(X.shape[0]):
            x = url_regex.sub(r'\g<1>\g<3>', X[i])
            x = media_regex.sub(r'\g<1>\g<3>', x)
            result[i] = x
        return result


if __name__ == '__main__':
    df = pd.DataFrame({'bio': ["""
i am just %URL% copy pasting my answer to an already answered question
Shubham %URL% Bhardwaj's answer to What life lessons are counter-intuitive or go against common sense or wisdom?
%URL%
%MEDIA%
ACTIONS %MEDIA% LIE LOUDER THAN WORDS:
I have grown up listening to “ACTIONS SPEAK LOUDER THAN WORDS” whole my life. 

""", """
i am just copy pasting my answer to an already answered question
%MEDIA%
%MEDIA%
        """, ""]})

    extractor = UrlAndMediaTextExtractor()
    stripper = UrlAndMediaTextStripper()
    mapper = sklearn_pandas.DataFrameMapper([
        ('bio', extractor),
        ('bio', stripper, {'alias': 'bio_stripped'})
    ])
    print(mapper.transform(df))
    print(mapper.transformed_names_)
