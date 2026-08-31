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
import pandas as pd
from xgboost import XGBClassifier

def build_classifier(trial: Trial | None, model: str) -> BaseEstimator:
    """ Build a scikit-learn classification model
        Args:
            trial: Optuna Trial.
            model: Scikit-learn classification model name.
        Returns:
            Scikit-learn model.
    """
    if model=="Logistic Regression":
        if trial is None:
            return LogisticRegression(max_iter=500, class_weight="balanced")

        params={"solver": trial.suggest_categorical("lr_solver", ["lbfgs", "liblinear", "newton-cg", "sag"]),
                "C": trial.suggest_float("lr_C",  0.1, 40, log=True)}
        if params["solver"]=="liblinear":
            params["penalty"]=trial.suggest_categorical("lr_penalty",  ["l1","l2"])

        return LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced",**params)

    elif model=="K-Nearest Neighbors":
        if trial is None:
            return KNeighborsClassifier(n_neighbors=5, weights= "uniform", metric="minkowski")

        params={"weights": "uniform",
                "metric": "minkowski",
                "n_neighbors": trial.suggest_int("knn_n_neighbors", 2, 25),
                "p": trial.suggest_int("k_nn_p",1,6)}
        
        return KNeighborsClassifier(**params)

    elif model=="Support Vector Machine":
        if trial is None:
            return SVC(kernel="rbf", probability=True, class_weight="balanced",random_state=RANDOM_STATE)

        params={"C": trial.suggest_float("svc_C",1e-3,100,log=True),
                "kernel":trial.suggest_categorical("svc_kernel",["linear","rbf","poly"])}
                
        if params["kernel"] in ["rbf","poly"]:
            params["gamma"]=trial.suggest_float("svc_gamma",1e-3,5,log=True)
        if params["kernel"] == ["poly"]:
            params["degree"]=trial.suggest_int("svc_degree",2,6)

        return SVC(probability=True,random_state=RANDOM_STATE, class_weight="balanced",**params)

    elif model=="Decision Tree":
        if trial is None:
            return DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE, class_weight="balanced")

        params={"criterion": trial.suggest_categorical("dt_criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("dt_max_depth", 2, 10),
                "min_samples_split": trial.suggest_int("dt_min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("dt_min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("dt_max_features", [None,"sqrt","log2"])}

        return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced",**params)

    elif model=="Random Forest":
        if trial is None:
            return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")

        params={"n_estimators": trial.suggest_int("rf_n_estimators", 500, 1500, step=100),
                "criterion": trial.suggest_categorical("rf_criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("rf_max_depth", 2, 15),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 25),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("rf_max_features", ["sqrt","log2", None]),
                "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False])}

        return RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced",**params)

    elif model=="XGBoost":
        if trial is None:
            return XGBClassifier(objective= "binary:logistic", eval_metric="auc", random_state=RANDOM_STATE, n_estimators =1000 ,n_jobs=1)

        params={"n_estimators": trial.suggest_int("xgb_n_estimators",500,1500, step=100),
                "scale_pos_weight": trial.suggest_float("xgb_scale_pos_weight", 1., 4.5),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.1, log=True),
                "max_depth": trial.suggest_int("xgb_max_depth", 2, 20),
                "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
                "gamma": trial.suggest_float("xgb_gamma", 0, 30),
                "subsample": trial.suggest_float("xgb_subsample", 0.35, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 5, log=True),
                "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 5, log=True)}

        return XGBClassifier(objective= "binary:logistic", eval_metric="auc", random_state=RANDOM_STATE,n_jobs=1,**params)

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

    transformers=[("encoder_binary", OneHotEncoder(drop="if_binary"), BINARY_FEATURES),
        ("encoder_categorical", OneHotEncoder(), CATEGORICAL_FEATURES)]

    if include_scaler is True:
        transformers.append(("scaler_numerical", StandardScaler(), NUMERICAL_FEATURES))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")
    
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

    return Pipeline(steps=[("preprocessor", preprocessor),("classifier", classifier)])


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
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    return scores.mean()