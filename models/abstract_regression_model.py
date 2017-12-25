"""
Represents a regression model fit on specified data with specified preprocessing steps.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
matplotlib.rcParams['grid.linestyle'] = ':'
matplotlib.rcParams['grid.alpha'] = 0.8

font = {'fontname': 'Palatino Linotype', 'fontsize': 11}
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Palatino'
matplotlib.rcParams['font.size'] = 11


class AbstractRegressionModel:
    def __init__(self, train, test, output_prefix):
        self.output_prefix = output_prefix
        self.train = train
        self.test = test

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        self.train_prediction = None
        self.test_prediction = None

    def fit(self):
        raise NotImplementedError()

    def save(self):
        joblib.dump(self.model, 'outputs/models/%s.bin' % self.output_prefix)

    def load(self):
        self.model = joblib.load('outputs/models/%s.bin' % self.output_prefix)

    def run(self, do_lowess=True, r_type='raw'):
        self.fit()
        self.stats()
        self.qq_plot()
        self.plot_predicted_vs_actual(do_lowess=do_lowess)
        # self.plot_residuals(do_lowess=False, r_type=r_type)
        self.plot_feature_importance()
        self.save()

    def stats(self):
        self.train_prediction = self.model.predict(X=self.X_train)
        self.test_prediction = self.model.predict(X=self.X_test)

        print('Stats (train | test):')
        print('\tR^2 score:\t\t%.4f\n\t\t\t\t\t%.4f' % (r2_score(self.y_train, self.train_prediction),
                                                        r2_score(self.y_test, self.test_prediction)))
        print('\tRMSE:\t\t\t%.4f\n\t\t\t\t\t%.4f' % (mean_squared_error(self.y_train, self.train_prediction) ** 0.5,
                                                     mean_squared_error(self.y_test, self.test_prediction) ** 0.5))
        print('\tMean error:\t\t%.4f\n\t\t\t\t\t%.4f' % (mean_absolute_error(self.y_train, self.train_prediction),
                                                         mean_absolute_error(self.y_test, self.test_prediction)))
        print()

    def plot_predicted_vs_actual(self, do_lowess=True, alpha=0.4):
        print('Plotting predicted vs. actual ...', end='')
        fig = plt.figure(figsize=(4.72441, 3.54331)) # size for the paper, can be anything
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(self.y_test, self.test_prediction, s=4, alpha=alpha, rasterized=True)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=1, rasterized=True)

        # plot lowess on top of it
        if do_lowess:
            self.__lowess(self.y_test, self.test_prediction, color='b')

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        fig.tight_layout()
        plt.savefig('outputs/predicted-vs-actual/%s.png' % self.output_prefix, bbox_inches='tight')
        plt.gcf().clear()
        print('done\n')

    def qq_plot(self):
        print('Plotting QQ ...', end='')
        train_prediction = self.model.predict(X=self.X_train)
        residuals = self.y_train - train_prediction
        plt.figure(figsize=(8, 8))
        stats.probplot(residuals, dist='norm', plot=plt)
        plt.savefig('outputs/qq-plots/%s.png' % self.output_prefix, bbox_inches='tight')
        plt.gcf().clear()
        print('done\n')

    def plot_residuals(self, on_train=False, do_lowess=True, r_type='standardized'):
        print('Plotting residuals ...', end='')
        train_residuals = self.__residuals(self.X_train, self.y_train, self.train_prediction, r_type)
        test_residuals = self.__residuals(self.X_test, self.y_test, self.test_prediction, r_type)
        plt.figure()
        plt.scatter(self.y_test, test_residuals, c='g', s=10)
        if on_train:
            plt.scatter(self.y_train, train_residuals, c='b', s=10, alpha=0.25)
            plt.hlines(y=0, xmin=0, xmax=1, colors='k', linestyles='dashed', linewidth=1)
        else:
            plt.hlines(y=0, xmin=0, xmax=1, colors='k', linestyles='dashed', linewidth=1)

        # plot lowess on top of it
        if do_lowess:
            self.__lowess(self.y_test, test_residuals, color='g')
            if on_train:
                self.__lowess(self.y_train, train_residuals, color='b')

        plt.ylabel('Residuals')
        plt.xlabel('Actual')
        plt.savefig('outputs/residuals/%s.png' % self.output_prefix, bbox_inches='tight')
        plt.gcf().clear()
        print('done\n')

    def plot_feature_importance(self):
        if not hasattr(self.model.named_steps['estimate'], 'feature_importances_') and not hasattr(self.model.named_steps['estimate'], 'feature_importances'):
            return
        if hasattr(self.model.named_steps['estimate'], 'feature_importances_'):
            importances = self.model.named_steps['estimate'].feature_importances_
        else:
            importances = self.model.named_steps['estimate'].feature_importances(self.model.named_steps['prepare_features'].fit_transform(self.X_train), self.y_train)
            print(importances)

        print('Plotting feature importances ...', end='')
        # can use specified feature names to generate fancy plots for the paper
        # feature_names = self.get_quora_feature_names()
        feature_names = self.get_facebook_feature_names()
        #feature_names = self.model.named_steps['prepare_features'].get_feature_names()

        df_features = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        df_features.sort_values(by='importance', ascending=True, inplace=True)
        plt.figure()
        df_features.plot(kind='barh', x='feature', y='importance', sort_columns=False, legend=False,
                         figsize=(4.6, 6))  # NB: quora-suitable size for the paper: figsize=(5.9, 12)
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')

        plt.savefig('outputs/importance/%s.png' % self.output_prefix, bbox_inches='tight')
        plt.gcf().clear()
        print('done\n')

    def get_quora_feature_names(self):
        feature_names = []
        for name in self.model.named_steps['prepare_features'].get_feature_names():
            if name.startswith('time_features') or (name.startswith('user_features') and 'presense' not in name) \
                    or 'question_fetched_answers_num' in name:
                name = 'F1: ' + name.split('__')[1]
            elif name.startswith('answer_features'):
                name = 'F2: ' + name.split('__')[1]
            elif 'subjectivity' in name or ('presense' in name and not 'topic' in name):
                name = 'F3: ' + name.split('__')[1]
            else:
                name = 'F4: ' + name.split('__')[1]
            feature_names.append(name.replace('_', '\_'))
        return feature_names

    def get_facebook_feature_names(self):
        feature_names = []
        for name in self.model.named_steps['prepare_features'].get_feature_names():
            if 'comment' in name:
                name = 'F4: ' + name.split('__')[1]
            elif 'mention' in name:
                name = 'F2: ' + name.split('__')[1]
            elif 'post' in name or name in ['life_events_num', 'media_to_all_normal_ratio']:
                name = 'F3: ' + name.split('__')[1]
            else:
                name = 'F1: ' + name.split('__')[1]
            feature_names.append(name.replace('_', '\_'))
        return feature_names

    def __residuals(self, X, y, y_predicted, r_type):
        """
        * 'raw' will return the raw residuals.
        * 'standardized' will return the standardized residuals, also known as
          internally studentized residuals, which is calculated as the residuals
          divided by the square root of MSE (or the STD of the residuals).
        * 'studentized' will return the externally studentized residuals, which
          is calculated as the raw residuals divided by sqrt(LOO-MSE * (1 -
          leverage_score)).
        """
        # make sure value of parameter 'r_type' is one we recognize
        assert r_type in ('raw', 'standardized', 'studentized'), ("Invalid option for 'r_type': {0}".format(r_type))

        y_true = y
        assert y_true.shape == y_predicted.shape, \
            ("Dimensions of y_true {0} do not match y_pred {1}".format(y_true.shape, y_predicted.shape))

        residuals = np.array(y_predicted - y_true)
        if r_type == 'standardized':
            residuals = residuals / np.std(residuals)
        elif r_type == 'studentized':
            # prepare a blank array to hold studentized residuals
            studentized_residuals = np.zeros(y_true.shape[0], dtype='float')

            # calculate hat matrix of X values so you can get leverage scores
            hat_matrix = np.dot(
                np.dot(X, np.linalg.inv(np.dot(np.transpose(X), X))),
                np.transpose(X)
            )
            # for each point, calculate studentized residuals w/ leave-one-out MSE
            for i in range(y_true.shape[0]):
                # make a mask so you can calculate leave-one-out MSE
                mask = np.ones(y_true.shape[0], dtype='bool')
                mask[i] = False
                loo_mse = np.average(residuals[mask] ** 2, axis=0)  # Leave-one-out MSE
                # calculate studentized residuals
                studentized_residuals[i] = residuals[i] / np.sqrt(loo_mse * (1 - hat_matrix[i, i]))
            residuals = studentized_residuals
        return residuals

    def __lowess(self, prediction, residuals, color):
        source_data = zip(prediction, residuals)
        xl, yl = [], []
        for x, y in sorted(source_data, key=lambda t: t[0]):
            xl.append(x)
            yl.append(y)
        lowess = sm.nonparametric.lowess(yl, xl)
        plt.plot(lowess[:, 0], lowess[:, 1], c=color, alpha=0.2)
