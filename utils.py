import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import math


class YellowModel:
    def __init__(self, name):
        self.name = name
        self.models = {'decisionTree': tree.DecisionTreeRegressor(),
                       'GaussianNB': GaussianNB(),
                       'Ridge': linear_model.Ridge(),
                       'SVR': svm.SVR(),
                       'linearRegression': LinearRegression(),
                       'randomForest': RandomForestRegressor()
                       }
        self._params = {
            'decisionTree': {
                'max_depth':  np.linspace(2, 10, 5),
                'min_samples_split': np.linspace(0.1, 1.0, 5),
                'min_samples_leaf': np.linspace(0.1, 0.5, 5)
            },
            'GaussianNB': None,
            'Ridge': {
                'normalize': [False, True],
                'alpha': np.linspace(0.0, 1.0, 10)
            },
            'SVR': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'gamma': [0.001, 0.01, 0.1, 1],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'linearRegression': {
                'normalize': [False, True]
            },
            'randomForest': {
                'n_estimators': [10, 20],
                'max_depth': np.linspace(2, 10, 5),
                'min_samples_split': np.linspace(0.1, 1.0, 5),
                'min_samples_leaf': np.linspace(0.1, 1.0, 5),
                'bootstrap': [True, False],
            }}
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.modes = ['one-hot', 'mean', 'compined']
        self.columns = None
        self.model = None
        self.confidense = None

    @staticmethod
    def __etl_data(path):
        data = pd.read_csv(path)
        data = data[['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY']]
        data['YALL'] = data.apply(lambda row: row['HY'] + row['AY'], axis=1)
        return data

    @staticmethod
    def __preprocess_one_hot_method(data):
        pd_home = pd.get_dummies(data['HomeTeam'], prefix='home')
        data = pd.concat([data, pd_home], axis=1)
        pd_away = pd.get_dummies(data['AwayTeam'], prefix='away')
        data = pd.concat([data, pd_away], axis=1)
        referee = pd.get_dummies(data['Referee'], prefix='referee')
        data = pd.concat([data, referee], axis=1)
        return data

    @staticmethod
    def __split_data(data, test_size):
        y = data[['YALL']].values
        x = data.drop(columns=['YALL']).values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test

    def set_data(self, path, mode, test_size):
        data = self.__etl_data(path)
        self.mode = mode
        if mode == 'one-hot':
            self.__set_data_one_hot(data, test_size)
        elif mode == 'mean':
            self.__set_data_mean(data, test_size)
        elif mode == 'compined':
            self.__set_data_compined(data, test_size)

    def __set_data_compined(self, data, test_size):
        data = self.__preprocess_one_hot_method(data)
        data_train, data_test = train_test_split(data, test_size=test_size)
        self.y_train = data_train[['YALL']].values
        self.y_test = data_test[['YALL']].values
        home_mean = dict(self._mean(data_train, 'HomeTeam', 'HY'))
        away_mean = dict(self._mean(data_train, 'AwayTeam', 'AY'))
        referee_mean = dict(self._mean(data_train, 'Referee', 'YALL'))
        away_temp = data_train[['HomeTeam', 'HY']].rename(columns={'HomeTeam': 'Team', 'HY': 'Y'})
        home_temp = data_train[['AwayTeam', 'AY']].rename(columns={'AwayTeam': 'Team', 'AY': 'Y'})
        compinedYellow = pd.concat([away_temp, home_temp], ignore_index=True)
        compined_mean = dict(self._mean(compinedYellow, 'Team', 'Y'))

        data_train['mean_home'] = data_train.apply(lambda row: home_mean[row['HomeTeam']], axis=1)
        data_test['mean_home'] = data_test.apply(lambda row: home_mean[row['HomeTeam']], axis=1)

        data_train['mean_away'] = data_train.apply(lambda row: away_mean[row['AwayTeam']], axis=1)
        data_test['mean_away'] = data_test.apply(lambda row: away_mean[row['AwayTeam']], axis=1)

        data_train['mean_total_home'] = data_train.apply(lambda row: compined_mean[row['HomeTeam']], axis=1)
        data_test['mean_total_home'] = data_test.apply(lambda row: compined_mean[row['HomeTeam']], axis=1)
        data_train['mean_total_away'] = data_train.apply(lambda row: compined_mean[row['AwayTeam']], axis=1)
        data_test['mean_total_away'] = data_test.apply(lambda row: compined_mean[row['AwayTeam']], axis=1)

        data_train['mean_ref'] = data_train.apply(lambda row: referee_mean[row['Referee']], axis=1)
        data_test['mean_ref'] = data_test.apply(lambda row: referee_mean[row['Referee']], axis=1)
        data_train = data_train.drop(columns=['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY', 'YALL'])
        data_test = data_test.drop(columns=['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY', 'YALL'])
        self.columns = data_train.columns
        self.x_train = data_train.values
        self.x_test = data_test.values
        pass

    @staticmethod
    def _mean(data, group_col, mean_col):
        return data.groupby(group_col).apply(lambda x: x[mean_col].mean())

    def __set_data_mean(self, data, test_size):
        data_train_final = pd.DataFrame()
        data_test_final = pd.DataFrame()
        data_train, data_test = train_test_split(data, test_size=test_size)
        self.y_train = data_train[['YALL']].values
        self.y_test =data_test[['YALL']].values
        home_mean = dict(self._mean(data_train, 'HomeTeam', 'HY'))
        away_mean = dict(self._mean(data_train, 'AwayTeam', 'AY'))
        referee_mean = dict(self._mean(data_train, 'Referee', 'YALL'))
        away_temp = data_train[['HomeTeam', 'HY']].rename(columns={'HomeTeam': 'Team', 'HY': 'Y'})
        home_temp = data_train[['AwayTeam', 'AY']].rename(columns={'AwayTeam': 'Team', 'AY': 'Y'})
        compinedYellow = pd.concat([away_temp, home_temp], ignore_index=True)
        compined_mean = dict(self._mean(compinedYellow, 'Team', 'Y'))

        data_train_final['mean_home'] = data_train.apply(lambda row: home_mean[row['HomeTeam']], axis=1)
        data_test_final['mean_home'] = data_test.apply(lambda row: home_mean[row['HomeTeam']], axis=1)

        data_train_final['mean_away'] = data_train.apply(lambda row: away_mean[row['AwayTeam']], axis=1)
        data_test_final['mean_away'] = data_test.apply(lambda row: away_mean[row['AwayTeam']], axis=1)

        data_train_final['mean_total_home'] = data_train.apply(lambda row: compined_mean[row['HomeTeam']], axis=1)
        data_test_final['mean_total_home'] = data_test.apply(lambda row: compined_mean[row['HomeTeam']], axis=1)
        data_train_final['mean_total_away'] = data_train.apply(lambda row: compined_mean[row['AwayTeam']], axis=1)
        data_test_final['mean_total_away'] = data_test.apply(lambda row: compined_mean[row['AwayTeam']], axis=1)

        data_train_final['mean_ref'] = data_train.apply(lambda row: referee_mean[row['Referee']], axis=1)
        data_test_final['mean_ref'] = data_test.apply(lambda row: referee_mean[row['Referee']], axis=1)

        self.columns = data_train.columns
        self.x_train = data_train_final.values
        self.x_test = data_test_final.values
        pass

    def __set_data_one_hot(self, data, test_size):
        data = self.__preprocess_one_hot_method(data)
        data = data.drop(columns=['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY'])
        self._params['decisionTree']['max_features'] = list(range(1, data.shape[1]))
        self._params['randomForest']['max_features'] = list(range(1, data.shape[1]))
        self.x_train, self.y_train, self.x_test, self.y_test = self.__split_data(data, test_size)
        self.columns = data.drop(columns=['YALL']).columns
        pass

    def fit(self, path, mode, test_size, model, params, times):
        mse_train_main = []
        mse_test_main = []
        for _ in range(times):
            try:
                self.set_data(path, mode, test_size)
                if not params:
                    clf = model
                    clf.fit(self.x_train, self.y_train)
                    y_predict_train = clf.predict(self.x_train)
                    y_predict_test = clf.predict(self.x_test)
                else:
                    clf = GridSearchCV(model, params, cv=10)
                    clf.fit(self.x_train, self.y_train)
                    y_predict_train = clf.best_estimator_.predict(self.x_train)
                    y_predict_test = clf.best_estimator_.predict(self.x_test)
            except:
                print('train continued due to absence of some factors.')
                continue
            mse_train = mean_squared_error(self.y_train, y_predict_train)
            mse_test = mean_squared_error(self.y_test, y_predict_test)
            mse_train_main.append(mse_train)
            mse_test_main.append(mse_test)
        return clf, np.mean(mse_train_main), np.mean(mse_test_main)

    def fit_all(self, path, test_size, times):
        best_mse_test = float('Inf')
        for mode in self.modes:
            print("====" * 20)
            print("mode : %s" % mode)
            for model_key, model in self.models.items():
                print("====" * 5)
                print("\tmodel: %s" % model_key)
                params = self._params[model_key]
                # TODO: check which clf to give back???
                clf, mse_train, mse_test = self.fit(path, mode, test_size, model, params, times)
                if mse_test < best_mse_test:
                    best_clf = clf
                    best_mse_test = mse_test
                    best_mse_train = mse_train
                    mode_best = self.mode
                    best_columns = self.columns
                print("\ttrain error: %s" % mse_train)
                print("\ttest error: %s" % mse_test)

        print("====" * 10)
        print("====" * 10)
        print("\t\ttrain error best per mode: %s" % best_mse_train)
        print("\t\ttest error best per mode: %s" % best_mse_test)
        print("\t\tmode best per model: %s" % mode_best)
        if hasattr(best_clf, 'best_estimator_'):
            print("\t\tmodel type best: %s" % str(best_clf.best_estimator_))
            joblib.dump([best_clf.best_estimator_, best_columns, math.sqrt(best_mse_test)], 'best_estimator_%s_%s.pkl' % (mode_best, self.name))
        else:
            print("\t\tmodel type best: %s" % str(best_clf))
            joblib.dump([best_clf, best_columns, math.sqrt(best_mse_test)], 'best_estimator_%s_%s.pkl' % (mode_best, self.name))
        print("====" * 10)
        print("====" * 10)
        return best_clf

    def load_model(self, path):
        model, best_columns, confidence = joblib.load(path + "_" + self.name + ".pkl")
        self.columns = best_columns
        self.model = model
        self.confidense = confidence

    def predict(self, team1, team2, referee):
        match = pd.get_dummies(pd.DataFrame({'home': [team1], 'away': [team2], 'referee': [referee]}))
        match = match.reindex(columns=self.columns, fill_value=0)
        yellows = self.model.predict(match.values)
        return yellows


def main_train(path):
    split_size = 0.1
    average_times = 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelobj = YellowModel('England')
        _ = modelobj.fit_all(path, split_size, average_times)

def main_train_global(path):
    pass


if __name__ == '__main__':
    path = './data/England_2018.csv'
    main_train(path)
