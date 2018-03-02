import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from datetime import datetime

CONFUSION_MATRIX = "confusion_matrix"
ACCURACY = "accuracy"


def k_fold_cv(x, y, model, k):
    kf = KFold(n_splits=k)
    confusion_mat = np.zeros(shape=(19, 19))
    accuracy_list = []
    print(model)
    for train_index, test_index in kf.split(x):
        print(datetime.now())
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_test = x.iloc[test_index]
        y_test = y.iloc[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        confusion_mat = confusion_mat + confusion_matrix(y_test, predictions)
        accuracy_list.append(accuracy_score(y_test, predictions))
    return {
        CONFUSION_MATRIX: confusion_mat,
        ACCURACY: accuracy_list,
    }
