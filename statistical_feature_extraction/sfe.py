# coding=utf-8
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from statistical_feature_extraction.test import protocol
from statistical_feature_extraction.test import executor

reload_data = False
save_info = False
np.random.seed(9231066)

_feature_vector_size = 1395

_columns = list(range(0, _feature_vector_size))
# noinspection PyTypeChecker
_columns.append('subject')
# noinspection PyTypeChecker
_columns.append('activity')

_nominal_columns = [_columns[-2], _columns[-1]]

MODELS = {
    "LOGISTIC_REGRESSION": 0,
    "RANDOM_FOREST": 1,
    "SVC": 2,
    "LINEAR_SVC": 3,
    "KNN": 4,
    "GAUSSIAN_NB": 5,
    "MLP_CLASSIFIER": 6,
    "DECISION_TREE": 7,
    "GAUSSIAN_PROCESS": 8
}


class HAR:
    def __init__(self):
        self.__data = None  # type: pd.DataFrame
        self.__models = [
            {
                "model": LogisticRegression()
            }, {
                "model": RandomForestClassifier()
            }, {
                "model": SVC()
            }, {
                "model": LinearSVC()
            }, {
                "model": KNeighborsClassifier(n_neighbors=7)
            }, {
                "model": GaussianNB()
            }, {
                "model": MLPClassifier()
            }, {
                "model": DecisionTreeClassifier()
            }, {
                "model": GaussianProcessClassifier()
            }
        ]
        self.__selected_models = []
        self.__train_x = None
        self.__train_y = None
        self.__test_x = None
        self.__test_y = None

    def load_data(self):
        root_dir = '.\\data\\'
        self.__data = pd.DataFrame()
        for dir_name, subdir_list, file_list in os.walk(root_dir):
            data_list = list()
            for f in file_list:
                sensors_data = pd.read_csv(dir_name + os.sep + f, header=None)
                sensors_data = _feature_extraction(sensors_data)
                sensors_data.set_value(_feature_vector_size, dir_name.split(os.sep)[3])
                sensors_data.set_value(_feature_vector_size + 1, dir_name.split(os.sep)[2])
                sensors_data = sensors_data.to_frame().T
                sensors_data.columns = _columns
                data_list.append(sensors_data)
                self.__data = self.__data.append(sensors_data)

    def save_pickle(self, path):
        self.__data.to_pickle(path)

    def load_pickle(self, path):
        self.__data = pd.read_pickle(path)

    def save_csv(self, path):
        dd = self.__data.reset_index(drop=True)
        dd.to_csv(path)

    def load_csv(self, path):
        self.__data = pd.read_csv(path)

    def save_hdf5(self, path):
        self.__data.to_hdf(path, key='HAR__data')

    def load_hdf5(self, path):
        self.__data = pd.read_hdf(path, key='HAR__data')

    def shuffle(self):
        self.__data = self.__data.sample(frac=1).reset_index(drop=True)

    def normalize(self):
        min_max_scaler = MinMaxScaler()
        y = self.__data[_nominal_columns]
        x = self.__data.drop(_nominal_columns, axis=1)
        x = min_max_scaler.fit_transform(x)  # type: pd.DataFrame
        self.__data = pd.DataFrame(x).join(y)

    def dimensionality_reduction(self, new_dimension):
        pca = PCA(n_components=new_dimension)
        y = self.__data[_nominal_columns]
        x = self.__data.drop(_nominal_columns, axis=1)
        x = pca.fit_transform(x)  # type: pd.DataFrame
        self.__data = pd.DataFrame(x).join(y)

    def drop_extra_features(self):
        y = self.__data[['activity']]
        x = self.__data.drop(_nominal_columns, axis=1)
        self.__data = pd.DataFrame(x).join(y)

    def split_data(self, test_size=0.0):
        self.__train_x, self.__test_x = train_test_split(self.__data, test_size=test_size)
        self.__train_y = self.__train_x[['activity']]
        self.__train_x = self.__train_x.drop('activity', axis=1)
        self.__test_y = self.__test_x[['activity']]
        self.__test_x = self.__test_x.drop('activity', axis=1)

    def select_models(self, models):
        del self.__selected_models[:]
        for model in models:
            if model in MODELS.values() and model not in self.__selected_models:
                self.__selected_models.append(model)

    def train(self):
        for model in self.__selected_models:
            self.__models[model]["model"].fit(self.__train_x, self.__train_y)

    def test(self, strategy, result):
        test_result_list = list()

        for model in self.__selected_models:
            test_result_list.append(executor.test(strategy=strategy,
                                                  result=result,
                                                  x=self.__data.drop('activity', axis=1),
                                                  y=self.__data['activity'],
                                                  model=self.__models[model]["model"],
                                                  k=10))
        return test_result_list


def _feature_extraction(data):
    def nlargest_index(df, n):
        return df.nlargest(n).index.unique()[0:n]

    """
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
    # first 225 statistical features
    statistical = data.min()
    statistical = statistical.append(data.max(), ignore_index=True)
    statistical = statistical.append(data.mean(), ignore_index=True)
    statistical = statistical.append(data.skew(), ignore_index=True)
    statistical = statistical.append(data.kurtosis(), ignore_index=True)

    # FFT features
    fft = pd.DataFrame(np.fft.fft(data))
    fft_angle = fft.applymap(np.angle)
    fft = fft.applymap(np.abs)
    largest_values = pd.Series()
    largest_angles = pd.Series()
    largest_indices = pd.Series()
    for i in range(0, 45):
        five_largest_idx = nlargest_index(fft.ix[:, i].map(abs), 5)  # is map(abs) redundant?
        largest_indices = largest_indices.append(pd.Series(five_largest_idx),
                                                 ignore_index=True)
        five_largest = fft_angle.ix[five_largest_idx, i].T
        largest_angles = largest_angles.append(five_largest)
        five_largest = fft.ix[five_largest_idx, i].T
        largest_values = largest_values.append(five_largest)

    # Autocorrelation
    autocorrelation = pd.Series()
    autocorrelation = autocorrelation.append(data.apply(lambda col: col.autocorr(1), axis=0))
    for i in range(5, 51, 5):
        autocorrelation = autocorrelation.append(data.apply(lambda col: col.autocorr(i), axis=0))

    # Make result
    feature_vector = pd.Series()
    feature_vector = feature_vector.append(statistical)
    feature_vector = feature_vector.append(largest_values)
    feature_vector = feature_vector.append(largest_angles)
    feature_vector = feature_vector.append(largest_indices)
    feature_vector = feature_vector.append(autocorrelation)
    return feature_vector


def main():
    har = HAR()
    if reload_data:
        har.load_data()
        if save_info:
            har.save_pickle('./statistical_feature_extraction/sample5.pkl')
    else:
        har.load_pickle('./statistical_feature_extraction/sample5.pkl')
        # har.save_csv('./sample.csv')

    print('loaded')
    har.shuffle()
    har.normalize()
    har.dimensionality_reduction(30)
    # har.split_data(0.4)
    har.drop_extra_features()
    har.select_models([MODELS["GAUSSIAN_NB"]])
    # har.train()
    result_list = har.test(protocol.TEST_STRATEGIES["K_FOLD"],
                           protocol.TEST_RESULTS["CONFUSION_MATRIX"])
    for ii in range(0, 19):
        for jj in range(0, 19):
            print('{:3d}'.format(int(result_list[0][ii][jj])), end='  ')
        print()

    result_list = har.test(protocol.TEST_STRATEGIES["K_FOLD"],
                           protocol.TEST_RESULTS["ACCURACY"])
    print(np.mean(result_list), end=' ±')
    print(np.std(result_list))


if __name__ == "__main__":
    main()
