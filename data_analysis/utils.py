import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


def get_config(path):
    config = None
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_scores(X_test, y_test, model):
    """
    Calculates score metrics and returns them

    :param pd.Dataframe X_test: A dataframe containing feature data
    :param pd.Dataframe y_test: A dataframe containing target data
    :param model: Used to compute scores. Must provide a predict() function
    :return: accuracy_score, f1_score, precision_score, recall_score
    :rtype: list(array, array, array, array)
    """

    return [
        accuracy_score(y_test, model.predict(X_test)),
        f1_score(y_test, model.predict(X_test), average="macro"),
        precision_score(y_test, model.predict(X_test), average="macro"),
        recall_score(y_test, model.predict(X_test), average="macro"),
    ]