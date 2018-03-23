from statistical_feature_extraction.test.strategies import k_fold
from statistical_feature_extraction.test import constants


def test(strategy, x, y, model, k=None):
    if strategy == constants.REPEATED_RANDOM_SUB_SAMPLING:
        pass
    elif strategy == constants.K_FOLD:
        return k_fold.k_fold_cv(x=x,
                                y=y,
                                model=model,
                                k=k)
    elif strategy == constants.LEAVE_ONE_OUT:
        pass
    else:
        return None
