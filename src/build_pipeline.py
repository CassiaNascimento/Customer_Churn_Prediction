from config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

classifiers={'Logistic Regression':
                LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
            'K-Nearest Neighbors':
                KNeighborsClassifier(n_neighbors=5, metric='minkowski', n_jobs=-1),
            'Support Vector Machine':
                SVC(kernel='rbf', probability=True),
            'Decision Tree':
                DecisionTreeClassifier(criterion='gini', splitter='best', random_state=RANDOM_STATE),
            'Random Forest':
                RandomForestClassifier(n_estimators=100, criterion='gini', random_state=RANDOM_STATE)}

def build_pipeline(model: str) -> Pipeline:
    """ Build a scikit-learn classification pipeline

        Steps:
            1. Encode categorical and binary features with One Hot Encoder
            2. Apply standard scaler to numerical features
            3. Build pipeline for chosen model

        Args:
            model: Name of the classification model.
        Returns:
            Scikit-learn Pipeline.
        Raises:
            ValueError: If model name is invalid.
    """

    if model not in classifiers:
        raise ValueError(f"Invalid model. Available models: {list(classifiers.keys())}")

    preprocessor=ColumnTransformer(transformers=[('encoder_binary', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), BINARY_FEATURES),
                                       ('encoder_categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
                                       ('scaler_numerical', StandardScaler(), NUMERICAL_FEATURES)], remainder='drop')

    pipeline=Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifiers[model])])
    logger.info("Pipeline (encoder, scaler, classifier) created for model: %s.", model)

    return pipeline

#column_names=pipe.named_steps['preprocessor'].get_feature_names_out()

