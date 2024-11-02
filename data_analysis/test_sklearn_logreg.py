import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import tqdm
import git


import utils

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir

def run_logistic_regression():

    config = utils.get_config(PATH_TO_ROOT + "/data_analysis/logreg.yaml")
    data = load_breast_cancer(as_frame=True)

    confusion_matrices = []
    scorings = []
    feature_importances = []

    X = data.data
    y = np.where(data.target == 0, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config["seed"])

    # scaler = StandardScaler(with_mean=True, with_std=True)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    #model = LogisticRegression(random_state=config["seed"])
    #model.fit(X_train, y_train)

    pipe = Pipeline(
        steps=[
            ("transform", StandardScaler()),
            ("clf", LogisticRegression(random_state=config["seed"]))
            ])

    pipe.fit(X_train, y_train)

    cf = confusion_matrix(y_test, pipe.predict(X_test)).ravel()
    scores = utils.get_scores(X_test, y_test, pipe)
    feature_importance = (np.exp(pipe["clf"].coef_[0]))

    eval_df = pd.DataFrame(
             cf,
             index = ['tn', 'fp', 'fn', 'tp'],
             columns=["Confusion matrix logreg on cancer data"]
        )

    score_df = pd.DataFrame(
            scores,
            index=['Accuracy', 'f1-score', 'Precision', 'Recall'],
            columns=["Scores logreg on cancer data"]
         )

    feature_importance_df = pd.DataFrame(
        feature_importance,
        index = X.columns,
       columns=["Feature importance"]
    )

    return eval_df, score_df, feature_importance_df

if __name__ == "__main__":
    eval_df, score_df, feature_importance_df = run_logistic_regression()
    print(eval_df)
    print(score_df)
    print(feature_importance_df)