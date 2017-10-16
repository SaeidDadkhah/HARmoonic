import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from statistical_feature_extraction.test import protocol


def k_fold_cv(result_mode, x, y, model, k):
    if result_mode == protocol.TEST_RESULTS["CONFUSION_MATRIX"]:
        return __k_fold_confusion_matrix(x, y, model, k)
    elif result_mode == protocol.TEST_RESULTS["ACCURACY"]:
        return __k_fold_accuracy(x, y, model, k)
    else:
        return None


def __k_fold_accuracy(x, y, model, k):
    kf = KFold(n_splits=k)
    accuracy_list = []
    for train_index, test_index in kf.split(x):
        print('a')
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_test = x.iloc[test_index]
        y_test = y.iloc[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy_list.append(accuracy_score(y_test, predictions))
    return accuracy_list


def __k_fold_confusion_matrix(x, y, model, k):
    kf = KFold(n_splits=k)
    confusion_matrix_list = np.zeros(shape=(19, 19))
    for train_index, test_index in kf.split(x):
        print('a')
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_test = x.iloc[test_index]
        y_test = y.iloc[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        # confusion_matrix_list.append(confusion_matrix(y_test, predictions))
        confusion_matrix_list = confusion_matrix_list + confusion_matrix(y_test, predictions)
    return confusion_matrix_list
