from math import sqrt
from pprint import pprint
from time import time

import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, tpe, space_eval, STATUS_OK, STATUS_FAIL
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from models.abstract_regression_model import AbstractRegressionModel
from models.utils import timing


def custom_r2(estimator, X_test, y_test):
    predictions = estimator.predict(X_test)
    return r2_score(y_test, predictions)


class HyperoptModel(AbstractRegressionModel):
    """ Model to used when the hyperparameter optimization is required """

    def __init__(self, train, test, output_prefix, cv=3, max_evals=50, n_jobs=1):
        super().__init__(train, test, output_prefix)

        self.pipeline = None
        self.features = None
        self.space = None
        self.best_ = None
        self._raw_features = None

        self.cv = cv
        self.n_jobs = n_jobs
        self.max_evals = max_evals
        self.hyperopt_eval_num = 0

    def objective(self, params):
        """ objective function for the optimization process"""
        start_time = time()
        self.pipeline.set_params(**params)
        shuffle = KFold(n_splits=self.cv, shuffle=True)
        score = cross_val_score(self.pipeline, self.X_train, self.y_train,
                                cv=shuffle, scoring='neg_mean_squared_error', n_jobs=self.n_jobs)

        r2 = cross_val_score(self.pipeline, self.X_train, self.y_train,
                             cv=shuffle, scoring=custom_r2, n_jobs=self.n_jobs)
        self.hyperopt_eval_num += 1
        result = {
            'loss': sqrt(-score.mean()),
            'r2': r2.mean(),
            'cv_eval_time': time() - start_time,
            'status': STATUS_FAIL if np.isnan(score.mean()) else STATUS_OK,
            'params': params.copy()
        }
        print('[{0}/{1}]\tcv_eval_time={2:.2f} sec\tRMSE={3:.6f}\tR^2={4:.6f}'.format(
            self.hyperopt_eval_num, self.max_evals, result['cv_eval_time'], result['loss'], result['r2']
        ))
        return result

    @property
    def raw_features(self):
        return self._raw_features

    @raw_features.setter
    def raw_features(self, raw_features):
        self._raw_features = raw_features
        self.X_train = self.train[list(filter(lambda column: column in raw_features, self.train.columns))]
        self.X_test = self.test[list(filter(lambda column: column in raw_features, self.test.columns))]
        self.y_train = self.train['score']
        self.y_test = self.test['score']

    def run(self, do_lowess=True, r_type='raw', alpha=0.1):
        self.fit()
        self.stats()
        self.best_parameters()

        try:
            self.plot_predicted_vs_actual(do_lowess=do_lowess, alpha=alpha)
            self.plot_feature_importance()
            #self.qq_plot()
        except Exception:
            print('Could not create plots')

    def fit(self):
        print("Performing parameters optimization...")
        with timing():
            trials = Trials()
            self.best_ = fmin(self.objective, self.space, algo=tpe.suggest,
                              max_evals=self.max_evals, trials=trials, verbose=True)

            # save the hyperparameter at each iteration to a csv file
            param_values = [x['misc']['vals'] for x in trials.trials]
            param_values = [{key: value for key in x for value in x[key]} for x in param_values]
            param_values = [space_eval(self.space, x) for x in param_values]

            param_df = pd.DataFrame(param_values)
            param_df['cv_eval_time'] = [r.get('cv_eval_time') for r in trials.results]
            param_df['RMSE'] = trials.losses()
            param_df['R^2'] = [r.get('r2') for r in trials.results]
            param_df.index.name = 'Iteration'
            param_df.to_csv('./outputs/parameters/%s.csv' % self.output_prefix)
        print()

        best_parameters = space_eval(self.space, self.best_)
        self.pipeline.set_params(**best_parameters)
        self.model = self.pipeline.fit(self.X_train, self.y_train)

    def best_parameters(self):
        print("Best parameters set:")
        best_parameters = space_eval(self.space, self.best_)
        pprint(best_parameters)
        print()
