from statistical_feature_extraction.test import protocol
from statistical_feature_extraction.test.strategies import k_fold


def test(strategy, x, y, model, k=None):
    if strategy == protocol.TEST_STRATEGIES["REPEATED_RANDOM_SUB_SAMPLING"]:
        pass
    elif strategy == protocol.TEST_STRATEGIES["K_FOLD"]:
        return k_fold.k_fold_cv(x=x,
                                y=y,
                                model=model,
                                k=k)
    elif strategy == protocol.TEST_STRATEGIES["LEAVE_ONE_OUT"]:
        pass
    else:
        return None
