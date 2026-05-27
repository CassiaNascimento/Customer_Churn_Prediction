from config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd

def build_classifier(trial: Trial | None, model: str) -> BaseEstimator:
    """ Build a scikit-learn classification model
        Args:
            trial: Optuna Trial.
            model: Scikit-learn classification model name.
        Returns:
            Scikit-learn model.
    """
    if model=='Logistic Regression':
        if trial is None:
            return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced')

        params={'solver': trial.suggest_categorical('solver', ['saga']),
                'l1_ratio': trial.suggest_float('l1_ratio',0.,1.),
                'C': trial.suggest_float('C',  1e-3, 1000, log=True)}

        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced',**params)

    elif model=='K-Nearest Neighbors':
        if trial is None:
            return KNeighborsClassifier(n_neighbors=5, metric='minkowski', n_jobs=-1)

        params={"n_neighbors": trial.suggest_int("knn_n_neighbors", 3, 50),
                "weights": trial.suggest_categorical("knn_weights",["uniform", "distance"]),
                "metric": trial.suggest_categorical("knn_metric",["euclidean","manhattan","minkowski"]),
                "p": trial.suggest_int("knn_p",1,3)}

        return KNeighborsClassifier(n_jobs=-1, **params)

    elif model=='Support Vector Machine':
        if trial is None:
            return SVC(kernel='rbf', probability=True)

        params={"C": trial.suggest_float("svc_C",1e-3,100,log=True),
                "kernel": trial.suggest_categorical("svc_kernel",["linear","rbf","poly"]),
                "gamma": trial.suggest_float("svc_gamma",1e-5,10,log=True),
                "degree": trial.suggest_int("svc_degree",2,5)}

        return SVC(probability=True,**params)

    elif model=='Decision Tree':
        if trial is None:
            return DecisionTreeClassifier(criterion='gini', splitter='best',random_state=RANDOM_STATE, class_weight='balanced')

        params={"criterion": trial.suggest_categorical("dt_criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("dt_max_depth", 2, 50),
                "min_samples_split": trial.suggest_int("dt_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("dt_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("dt_max_features", [None,"sqrt","log2"])}

        return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced',**params)

    elif model=='Random Forest':
        if trial is None:
            return RandomForestClassifier(n_estimators=100, criterion='gini', random_state=RANDOM_STATE, class_weight='balanced')

        params={"n_estimators": trial.suggest_int("rf_n_estimators", 100, 1000, step=100),
                "criterion": trial.suggest_categorical("rf_criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("rf_max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("rf_max_features", ["sqrt","log2", None]),
                "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
                "n_jobs": -1}

        return RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced',**params)

    raise ValueError(f"Invalid model: {model}")

def build_preprocessor() -> ColumnTransformer:
    """ Build preprocessor
        Steps:
            1. Encode categorical and binary features with One Hot Encoder
            2. Apply standard scaler to numerical features
        Args:
            None
        Returns:
            Scikit-learn column transformer.
    """
    return ColumnTransformer(transformers=[
        ('encoder_binary', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), BINARY_FEATURES),
        ('encoder_categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ('scaler_numerical', StandardScaler(), NUMERICAL_FEATURES)], remainder='drop')
    return

def build_pipeline(trial: Trial | None, model: str) -> Pipeline:
    """ Build a scikit-learn classification pipeline

        Args:
            trial: Optuna Trial.
            model: Scikit-learn classification model name.
        Returns:
            Scikit-learn Pipeline.
    """
    preprocessor=build_preprocessor()
    classifier=build_classifier(trial,model)

    return Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifier)])


def objective(trial: Trial, X: pd.DataFrame, y: pd.Series, model: str, scoring: str = "recall", n_splits: int=5) -> float:
    """ Optuna objetive function

        Args:
            trial: Optuna Trial.
            X: Dataframe of features.
            y: Series of target values.
            model: Scikit-learn classification model name.

        Returns:
            Chosen score averaged between 5 k-folds (Default).
    """
    pipeline = build_pipeline(trial=trial, model=model)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return score.mean()
