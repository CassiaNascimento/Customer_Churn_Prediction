from config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO, force=True)

def build_pipeline(model):
    """ Build classification pipeline
        1. One Hot Enconder categorical and binary features
        2. Scale numerical features
        3. Build pipeline for chosen model
    """
    classifier={'Logistic Regression': LogisticRegression(max_iter=1000, random_state=0),
                'Random Forest': RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)}

    if model not in classifier:
        raise ValueError('This is not a valid model. Choose from: [\'Logistic Regression\', \'Random Forest\']')

    ct=ColumnTransformer(transformers=[('encoder_binary', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), BINARY_FEATURES),
                                       ('encoder_categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
                                       ('scaler_numerical', StandardScaler(), NUMERICAL_FEATURES)])

    pipe=Pipeline([('preprocessor', ct),('classifier', classifier[model])])
    logging.info(f"Pipeline (encoder, scaler, classifier) created.")

    return pipe

#column_names=pipe.named_steps['preprocessor'].get_feature_names_out()

