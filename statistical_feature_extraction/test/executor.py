from statistical_feature_extraction.test.strategy import leave_one_out
from statistical_feature_extraction.test.strategy import k_fold_cv
from statistical_feature_extraction.test import constants


def test(strategy, x, y, model, k=None, groups=None):
    if strategy == constants.REPEATED_RANDOM_SUB_SAMPLING:
        pass
    elif strategy == constants.K_FOLD:
        return k_fold_cv(x=x,
                         y=y,
                         model=model,
                         k=k)
    elif strategy == constants.LEAVE_ONE_OUT:
        return leave_one_out(x=x,
                             y=y,
                             model=model,
                             groups=groups)
    else:
        return None
