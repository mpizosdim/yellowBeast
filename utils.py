import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import math
import glob
import re


class GeneralModel:
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
                'max_depth': np.linspace(2, 10, 5),
                'min_samples_split': np.linspace(0.1, 1.0, 5),
                'min_samples_leaf': np.linspace(0.1, 0.5, 5)
            },
            'GaussianNB': {},
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
                'min_samples_leaf': np.linspace(0.1, 0.5, 5),
                'bootstrap': [True, False],
            }}
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.columns_one_hot = None
        self.model = None
        self.confidense = None

    @staticmethod
    def get_data(path, columns):
        if isinstance(path, list):
            list_ = []
            for file_ in path:
                year = re.search("_(\d+).csv", file_).group(1)
                df = pd.read_csv(file_)
                df['year'] = year
                list_.append(df)
            data = pd.concat(list_, axis=0, ignore_index=True)
        else:
            data = pd.read_csv(path)
        data = data[columns]
        data['YALL'] = data.apply(lambda row: row['HY'] + row['AY'], axis=1)
        return data

    def _set_data(self):
        pass

    def fit(self, path, test_size):
        best_clf, best_columns, confidence = self._fit_all(path, test_size, self.columns, self._set_data)
        self.best_clf = best_clf
        self.feature_columns = best_columns
        self.confidense = confidence

    def _fit(self, path, test_size, model, params, columns, setter_data):
        data = self.get_data(path, columns)
        while True:
            try:
                x_train, x_test, y_train, y_test, columns_one_hot = setter_data(data.sample(frac=1).reset_index(drop=True), test_size)
                self.columns_one_hot = columns_one_hot
                clf = GridSearchCV(model, params, cv=15, scoring=make_scorer(mean_squared_error, greater_is_better=False))
                clf.fit(x_train, y_train)
                y_predict_test = clf.best_estimator_.predict(x_test)
                mse_test = mean_squared_error(y_test, y_predict_test)
                mse_train = -clf.best_score_
                break
            except:
                print('reshuffle to find other values.')
        return clf, mse_train, mse_test

    def _fit_all(self, path, test_size, columns, setter_data):
        best_mse_test = float('Inf')
        for model_key, model in self.models.items():
            print("====" * 5)
            print("\tmodel: %s" % model_key)
            params = self._params[model_key]
            clf, mse_train, mse_test = self._fit(path, test_size, model, params, columns, setter_data)
            if mse_test < best_mse_test:
                best_clf = clf
                best_mse_test = mse_test
                best_mse_train = mse_train
                best_columns = self.columns_one_hot
            print("\ttrain error: %s" % mse_train)
            print("\ttest error: %s" % mse_test)
        print("====" * 10)
        print("====" * 10)
        print("\t\ttrain error best per mode: %s" % best_mse_train)
        print("\t\ttest error best per mode: %s" % best_mse_test)
        if hasattr(best_clf, 'best_estimator_'):
            print("\t\tmodel type best: %s" % str(best_clf.best_estimator_))
            return best_clf.best_estimator_, best_columns, math.sqrt(best_mse_test)
        else:
            print("\t\tmodel type best: %s" % str(best_clf))
            return best_clf, best_columns, math.sqrt(best_mse_test)


class LocalModel(GeneralModel):
    def __init__(self, name):
        super().__init__(name)
        self.columns = ['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY']

    @staticmethod
    def __preprocess_one_hot_method(data):
        pd_home = pd.get_dummies(data['HomeTeam'], prefix='home')
        data = pd.concat([data, pd_home], axis=1)
        pd_away = pd.get_dummies(data['AwayTeam'], prefix='away')
        data = pd.concat([data, pd_away], axis=1)
        referee = pd.get_dummies(data['Referee'], prefix='referee')
        data = pd.concat([data, referee], axis=1)
        return data

    def _set_data(self, data, test_size):
        data = self.__preprocess_one_hot_method(data)
        data = data.drop(columns=['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY'])
        x_train, y_train, x_test, y_test = self.__split_data(data, test_size)
        columns = data.drop(columns=['YALL']).columns
        return x_train, x_test, y_train, y_test, columns

    @staticmethod
    def __split_data(data, test_size):
        y = data[['YALL']].values
        x = data.drop(columns=['YALL']).values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test

    def load_model(self, path):
        model, best_columns, confidence = joblib.load(path)
        self.columns = best_columns
        self.model = model
        self.confidense = confidence

    def save_model(self):
        joblib.dump([self.best_clf, self.feature_columns, self.confidense],
                    'best_estimator_local_%s.pkl' % self.name)

    def predict(self, team1, team2, referee):
        match = pd.get_dummies(pd.DataFrame({'home': [team1], 'away': [team2], 'referee': [referee]}))
        match = match.reindex(columns=self.columns, fill_value=0)
        yellows = self.model.predict(match.values)
        return yellows


class GlobalModel(GeneralModel):
    def __init__(self, name):
        super().__init__(name)
        self.columns = ['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY', 'year']
        self.home_mean = None
        self.away_mean = None
        self.referee_mean = None

    @staticmethod
    def __mean(data, group_col, mean_col):
        return data.groupby(group_col).apply(lambda x: x[mean_col].mean())

    def _set_data(self, data, test_size):
        data_train_final = pd.DataFrame()
        data_test_final = pd.DataFrame()
        data_train, data_test = train_test_split(data, test_size=test_size)

        self.home_mean = dict(self.__mean(data_train, ['HomeTeam', 'year'], 'HY'))
        self.away_mean = dict(self.__mean(data_train, ['AwayTeam', 'year'], 'AY'))
        self.referee_mean = dict(self.__mean(data_train, ['Referee', 'year'], 'YALL'))
        away_temp = data_train[['HomeTeam', 'HY', 'year']].rename(columns={'HomeTeam': 'Team', 'HY': 'Y'})
        home_temp = data_train[['AwayTeam', 'AY', 'year']].rename(columns={'AwayTeam': 'Team', 'AY': 'Y'})
        compinedYellow = pd.concat([away_temp, home_temp], ignore_index=True)
        self.compined_mean = dict(self.__mean(compinedYellow, ['Team', 'year'], 'Y'))

        data_train_final['mean_home'] = data_train.apply(lambda row: self.home_mean[(row['HomeTeam'],row['year'])], axis=1)
        data_test_final['mean_home'] = data_test.apply(lambda row: self.home_mean[(row['HomeTeam'], row['year'])], axis=1)

        data_train_final['mean_away'] = data_train.apply(lambda row: self.away_mean[(row['AwayTeam'], row['year'])], axis=1)
        data_test_final['mean_away'] = data_test.apply(lambda row: self.away_mean[(row['AwayTeam'], row['year'])], axis=1)

        data_train_final['mean_total_home'] = data_train.apply(lambda row: self.compined_mean[(row['HomeTeam'], row['year'])], axis=1)
        data_test_final['mean_total_home'] = data_test.apply(lambda row: self.compined_mean[(row['HomeTeam'], row['year'])], axis=1)
        data_train_final['mean_total_away'] = data_train.apply(lambda row: self.compined_mean[(row['AwayTeam'], row['year'])], axis=1)
        data_test_final['mean_total_away'] = data_test.apply(lambda row: self.compined_mean[(row['AwayTeam'], row['year'])], axis=1)

        data_train_final['mean_ref'] = data_train.apply(lambda row: self.referee_mean[(row['Referee'], row['year'])], axis=1)
        data_test_final['mean_ref'] = data_test.apply(lambda row: self.referee_mean[(row['Referee'], row['year'])], axis=1)

        columns = data_train_final.columns
        x_train = data_train_final.values
        x_test = data_test_final.values
        y_train = data_train[['YALL']].values
        y_test = data_test[['YALL']].values
        return x_train, x_test, y_train, y_test, columns

    def load_model(self, path):
        model, best_columns, confidence, home_mean, away_mean, referee_mean, compined_mean = joblib.load(path)
        self.columns = best_columns
        self.model = model
        self.confidense = confidence
        self.home_mean = home_mean
        self.away_mean = away_mean
        self.referee_mean = referee_mean
        self.compined_mean = compined_mean

    def save_model(self):
        joblib.dump([self.best_clf, self.feature_columns, self.confidense, self.home_mean, self.away_mean, self.referee_mean, self.compined_mean],
                    'best_estimator_global_%s.pkl' % self.name)

    def predict(self, year, team1, team2, referee):
        input = np.array([self.home_mean[(team1, year)], self.away_mean[(team2, year)], self.compined_mean[(team1, year)], self.compined_mean[(team2, year)], self.referee_mean[(referee, year)]])
        yellows = self.model.predict(input.reshape(1, -1))[0]
        return yellows


class CompinedModel(GeneralModel):
    def __init__(self, name, globalmodel, localmodel):
        super().__init__(name)
        self.globalmodel = globalmodel
        self.localmodel = localmodel
        self.columns = ['HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY']

    def _set_data(self, data, test_size):
        prediction_global = data.apply(lambda row: self.globalmodel.predict('2018', row['HomeTeam'], row['AwayTeam'], row['Referee']), axis=1)
        prediction_local = data.apply(lambda row: self.localmodel.predict(row['HomeTeam'], row['AwayTeam'], row['Referee'])[0], axis=1)
        X = pd.concat([prediction_global, prediction_local], axis=1, ignore_index=True).values
        Y = data[['YALL']].values
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        columns = ['not usefull']
        return x_train, x_test, y_train, y_test, columns

    def load_model(self, path):
        model, confidence,  = joblib.load(path)
        self.model = model
        self.confidense = confidence

    def save_model(self):
        joblib.dump([self.best_clf, self.confidense],
                    'best_estimator_compined_%s.pkl' % self.name)

    def predict(self, year, team1, team2, referee):
        global_pred = self.globalmodel.predict(year, team1, team2, referee)
        local_pred = self.localmodel.predict(team1, team2, referee)[0]
        yellows = self.model.predict(np.array([global_pred, local_pred]).reshape(1, -1))[0]
        return yellows

# ========================================================================
# main
# ========================================================================


def main_train_global():
    split_size = 0.2
    paths = glob.glob('data/England*') + glob.glob('data/Scotland*')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelobj = GlobalModel('England_Scotland')
        modelobj.fit(paths, split_size)
        modelobj.save_model()


def main_train_local():
    split_size = 0.15
    path = 'data/England_2018.csv'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelob = LocalModel('England_2018')
        modelob.fit(path, split_size)
        modelob.save_model()


def main_train_compined():
    split_size = 0.15
    globalmodel = GlobalModel('England_Scotland')
    globalmodel.load_model("best_estimator_global_England_Scotland.pkl")
    localmodel = LocalModel('England')
    localmodel.load_model("best_estimator_local_England_2018.pkl")
    path = 'data/England_2018.csv'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelob = CompinedModel('England_compined', globalmodel, localmodel)
        modelob.fit(path, split_size)
        modelob.save_model()


def main_predict_global():
    modelobj = GlobalModel('England_Scotland')
    modelobj.load_model("best_estimator_global_England_Scotland.pkl")
    pred = modelobj.predict('2018', 'Cardiff', 'Wolves', 'A Marriner')
    print(pred)
    print(pred - modelobj.confidense)
    print(pred + modelobj.confidense)


def main_predict_local():
    modelobj = LocalModel('England')
    modelobj.load_model("best_estimator_local_England_2018.pkl")
    pred = modelobj.predict('Cardiff', 'Wolves', 'A Marriner')
    print(pred)
    print(pred - modelobj.confidense)
    print(pred + modelobj.confidense)


def main_predict_compined():
    globalmodel = GlobalModel('England_Scotland')
    globalmodel.load_model("best_estimator_global_mean_England_Scotland.pkl")
    localmodel = LocalModel('England')
    localmodel.load_model("best_estimator_local_one-hot_England_2018.pkl")
    modelcompined = CompinedModel('England_compined', globalmodel, localmodel)
    modelcompined.load_model("best_estimator_compined_England_compined.pkl")
    pred = modelcompined.predict('2018', 'Cardiff', 'Wolves', 'A Marriner')
    print(pred)
    print(pred - modelcompined.confidense)
    print(pred + modelcompined.confidense)


if __name__ == '__main__':
    main_train_global()
    main_train_local()
    main_train_compined()
    #main_predict_compined()
