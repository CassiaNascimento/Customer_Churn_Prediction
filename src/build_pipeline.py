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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

        params={'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
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
            return SVC(kernel='rbf', probability=True, class_weight='balanced')

        params={"C": trial.suggest_float("svc_C",1e-3,100,log=True),
                "kernel": trial.suggest_categorical("svc_kernel",["linear","rbf","poly"]),
                "gamma": trial.suggest_float("svc_gamma",1e-5,10,log=True),
                "degree": trial.suggest_int("svc_degree",2,5)}

        return SVC(probability=True, class_weight='balanced',**params)

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

    elif model=='XGBoost':
        if trial is None:
            return XGBClassifier(n_estimators =1000, random_state=RANDOM_STATE)

        params={"objective": "binary:logistic",  # or reg:squarederror
                "eval_metric": "auc",
                "booster": "gbtree",
                "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
                "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1, 20),
                "gamma": trial.suggest_float("xgb_gamma", 0, 10),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("xgb_colsample_bylevel", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 100, log=True),
                "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-8, 100, log=True),
                "scale_pos_weight": trial.suggest_float("xgb_scale_pos_weight", 0.5, 20),
                "tree_method": "hist",
                "n_jobs": -1}

        return XGBClassifier(n_estimators =1000, random_state=RANDOM_STATE,**params)

    elif model=='LightGBM':
        if trial is None:
            return LGBMClassifier(n_estimators =1000,random_state=RANDOM_STATE)

        params={"learning_rate": trial.suggest_float("lgbm_learning_rate", 0.005, 0.05, log=True),
                "max_depth": trial.suggest_int("lgbm_max_depth", 4, 10),
                "num_leaves": trial.suggest_int("lgbm_num_leaves", 16, 512),
                "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 20, 100),
                "subsample": trial.suggest_float("lgbm_subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 10, log=True)}

        return LGBMClassifier(n_estimators =300,random_state=RANDOM_STATE,**params)

    raise ValueError(f"Invalid model: {model}")

def build_preprocessor(include_scaler:bool=True) -> ColumnTransformer:
    """ Build preprocessor
        Steps:
            1. Encode categorical and binary features with One Hot Encoder
            2. Apply standard scaler to numerical features
        Args:
            None
        Returns:
            Scikit-learn column transformer.
    """
    if include_scaler:
        return ColumnTransformer(transformers=[
            ('encoder_binary', OneHotEncoder(drop='if_binary'), BINARY_FEATURES),
            ('encoder_categorical', OneHotEncoder(), CATEGORICAL_FEATURES),
            ('scaler_numerical', StandardScaler(), NUMERICAL_FEATURES)], remainder='passthrough')
    else:
        return ColumnTransformer(transformers=[
            ('encoder_binary', OneHotEncoder(drop='if_binary'), BINARY_FEATURES),
            ('encoder_categorical', OneHotEncoder(), CATEGORICAL_FEATURES)], remainder='passthrough')

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


def objective(trial: Trial, X: pd.DataFrame, y: pd.Series, model: str, scoring: str = "f1", n_splits: int=5) -> float:
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
