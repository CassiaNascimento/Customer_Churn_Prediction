from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

def evaluate_model(fitted_pipeline, X_test, y_test):
    """ Evaluate model performance

        Calculate the most significant scores to evaluate the classification model performance

        Args:
            fitted_pipeline: scikit-learn Pipeline
            X_test: matrix of test set features
            y_test: numpy array of test set target values

        Returns:
            Dictionaire of evaluation metrics
    """
    y_pred=fitted_pipeline.predict(X_test)
    y_prob = fitted_pipeline.predict_proba(X_test)[:,1]
    classifier_name=type(fitted_pipeline.named_steps['classifier']).__name__

    metrics={'classifier_name':classifier_name,
             'accuracy': accuracy_score(y_test, y_pred),
             'roc_auc': roc_auc_score(y_test, y_prob),
             'f1': f1_score(y_test, y_pred),
             'precision': precision_score(y_test, y_pred),
             'recall': recall_score(y_test, y_pred),
             'average_precision': average_precision_score(y_test, y_prob),
             'confusion_matrix': confusion_matrix(y_test, y_pred),
             'report': classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'], output_dict=False)}

    logger.info(f"Calculated accuracy, roc_auc, f_1, precision, recall, average_precision, confusion_matrix and report for {classifier_name} classification model.")

    return metrics
