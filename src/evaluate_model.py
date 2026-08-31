from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
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
             'accuracy': round(accuracy_score(y_test, y_pred),3),
             'f1': round(f1_score(y_test, y_pred),3),
             'precision': round(precision_score(y_test, y_pred),3),
             'recall': round(recall_score(y_test, y_pred),3),
             'roc_auc': round(roc_auc_score(y_test, y_prob),3),
             'average_precision': round(average_precision_score(y_test, y_prob),3),
             'confusion_matrix': confusion_matrix(y_test, y_pred)}
    report= classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'], output_dict=False)

    logger.info(f"Calculated accuracy, roc_auc, f_1, precision, recall, average_precision, confusion_matrix and report for {classifier_name} classification model.")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    plt.plot(fpr, tpr,color="red")
    plt.fill_between(fpr, tpr,color="red",alpha=0.2)
    plt.plot(np.linspace(0.,1,10), np.linspace(0.,1,10),"--",color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positie Rate")
    plt.title("ROC curve")
    plt.text(0.05, 0.95, f'AUC: {metrics["roc_auc"]}', fontsize=12,
        verticalalignment='top', bbox={'boxstyle':'round', 'facecolor':'white', 'alpha':0.5})
    plt.show()
    return metrics,report
