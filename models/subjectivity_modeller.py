from pprint import pprint
from time import time

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


class SubjectivityModel:
    def __init__(self):
        self.subj_data = []
        self.obj_data = []

        self.X, self.X_train, self.X_test = None, None, None
        self.y, self.y_train, self.y_test = None, None, None

        self.best_estimator = None
        self.best_params = None

        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier())
        ])

    def read_data(self):
        with open('../data/subjectivity/subjective_questions.txt', encoding='utf-8') as f:
            for line in f:
                if line.strip() != '':
                    self.subj_data.append(line.strip().lower())
        with open('../data/subjectivity/objective_questions.txt', encoding='utf-8') as f:
            for line in f:
                if line.strip() != '':
                    self.obj_data.append(line.strip().lower())

        self.X = np.array(self.subj_data + self.obj_data)
        self.y = np.array([1] * len(self.subj_data) + [0] * len(self.obj_data))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)

    def load_model(self, filename='../data/subjectivity/subj.clf'):
        self.best_estimator = joblib.load(filename)

    def train_and_save_full(self, filename='../data/subjectivity/subj.clf'):
        kwargs = {
            'clf__alpha': 1e-06,
            'clf__loss': 'modified_huber',
            'clf__n_iter': 50,
            'clf__penalty': 'elasticnet',
            'tfidf__norm': 'l1',
            'tfidf__use_idf': False,
            'vect__max_df': 0.95,
            'vect__max_features': None,
            'vect__ngram_range': (1, 2),
        }
        self.pipeline.set_params(**kwargs)
        self.pipeline.fit(self.X_train, self.y_train)
        joblib.dump(self.pipeline, filename)

    def train_bayes(self):
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('nb', MultinomialNB())
        ])

        parameters = {
            'vect__max_df': (0.5, 0.8, 0.95, 1.0),
            'vect__max_features': (None, 1000, 5000, 8000, 10000),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)),
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
        }

        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in self.pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(self.X_train, self.y_train)
        print("done in %0.3fs" % (time() - t0))
        print()

        self.best_estimator = grid_search.best_estimator_
        self.best_params = self.best_estimator.get_params()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, self.best_params[param_name]))

    def train_and_save(self, filename='../data/subjectivity/subj.clf'):
        parameters = {
            'vect__max_df': (0.5, 0.8, 0.95, 1.0),
            'vect__max_features': (None, 1000, 5000, 8000, 10000),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)),
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet', 'l1'),
            'clf__n_iter': (10, 50, 80),
            'clf__loss': ('log', 'modified_huber')
        }

        grid_search = GridSearchCV(self.pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in self.pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(self.X_train, self.y_train)
        print("done in %0.3fs" % (time() - t0))
        print()

        self.best_estimator = grid_search.best_estimator_
        self.best_params = self.best_estimator.get_params()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, self.best_params[param_name]))

        joblib.dump(self.best_estimator, filename)
        with open(filename + '.params.txt', 'w', encoding='utf-8') as f:
            f.write("Parameters:\n")
            pprint(parameters, f)
            f.write("\nBest parameters set:\n")
            for param_name in sorted(parameters.keys()):
                f.write("\t%s: %r\n" % (param_name, self.best_params[param_name]))

    def test(self):
        y_pred = self.best_estimator.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

    def check_errors(self):
        for q, y_true, y_predicted, (y_proba_0, y_proba_1) in zip(self.X_test, self.y_test, self.best_estimator.predict(self.X_test), self.best_estimator.predict_proba(self.X_test)):
            if y_true != y_predicted:
                print('true: %d, proba-1: %.4f\t\t%s' % (y_true, y_proba_1, q))


class SubjectivityClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        super(SubjectivityClassifier, self).__init__()
        self.filename = filename
        self.model = joblib.load(self.filename)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        result = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            result[i] = np.array([self.model.predict_proba([X[i]])[0][1]])
        return result


def main():
    model = SubjectivityModel()
    model.read_data()

    # model.train_bayes()
    # model.train_and_save()
    # model.load_model()
    # model.test()
    # model.check_errors()

    model.train_and_save_full()


if __name__ == '__main__':
    main()
