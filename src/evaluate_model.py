from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, roc_auc_score, f1_score, fbeta_score, precision_score, recall_score, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_model(fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5):
    """ Evaluate model performance

        Calculate metrics using custom threshold

        Args:
            fitted_pipeline: scikit-learn Pipeline
            X_test: matrix of test set features
            y_test: numpy array of test set target values
            threshold: probability threshold for positive class classification

        Returns:
            Tuple of metrics dictionarie and classification report
    """
    y_prob = fitted_pipeline.predict_proba(X_test)[:,1]
    y_pred=(y_prob>=threshold).astype(int)
    classifier_name=type(fitted_pipeline.named_steps['classifier']).__name__

    metrics={
            'classifier_name':classifier_name,
            'threshold': round(threshold,3),
            'accuracy': round(accuracy_score(y_test, y_pred),3),
            'f1': round(f1_score(y_test, y_pred),3),
            'f2': round(fbeta_score(y_test, y_pred, beta=2),3),
            'precision': round(precision_score(y_test, y_pred),3),
            'recall': round(recall_score(y_test, y_pred),3),
            'roc_auc': round(roc_auc_score(y_test, y_prob),3),
            'average_precision': round(average_precision_score(y_test, y_prob),3),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()}

    report= classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'], output_dict=False)

    logger.info(f"Calculated accuracy, f_1, f_2, precision, recall, roc_auc, average_precision, confusion_matrix and report for {classifier_name} classification model (Threshold: {threshold}).")

    return metrics,report

def view_ROC_curve(fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series, save=False, save_path=None):
    """ ROC Curve and AUC

        Plot the ROC curve and evaluate the Area Under the ROC curve score for a fitted pipeline

        Args:
            fitted_pipeline: scikit-learn Pipeline
            X_test: matrix of test set features
            y_test: numpy array of test set target values

        Returns:
            Figure and axes
    """
    y_prob = fitted_pipeline.predict_proba(X_test)[:,1]
    roc_auc=round(roc_auc_score(y_test, y_prob),3)
    classifier_name=type(fitted_pipeline.named_steps['classifier']).__name__
    
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr,color="red",label=f'AUC: {roc_auc}')
    ax.fill_between(fpr, tpr,color="red",alpha=0.15)
    ax.plot([0, 1], [0, 1],"--",color="black")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve ({classifier_name})")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if save == True:
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved ROC Curve and AUC for {classifier_name} classification model.")
    else:
        plt.show()
        logger.info(f"ROC Curve and AUC for {classifier_name} classification model.")
    
    return fig, ax

def view_PR_curve(fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series, save=False, save_path=None):
    """ Precision Recall curve

        Plot the Precision Recall curve and evaluate AP score for a fitted pipeline

        Args:
            fitted_pipeline: scikit-learn Pipeline
            X_test: matrix of test set features
            y_test: numpy array of test set target values

        Returns:
            Figure and axes
    """
    y_prob = fitted_pipeline.predict_proba(X_test)[:,1]
    ap_score=round(average_precision_score(y_test, y_prob),3)
    classifier_name=type(fitted_pipeline.named_steps['classifier']).__name__
    
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob, pos_label=1)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(recalls, precisions,color="red",label=f'AP: {ap_score}')
    ax.fill_between(recalls, precisions,color="red",alpha=0.15)
    no_skill=len(y_test[y_test==1])/len(y_test)
    ax.plot([0, 1], [no_skill, no_skill],"--",color="black")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve ({classifier_name})")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if save == True:
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved Precision-Recall Curve for {classifier_name} classification model.")
    else:
        plt.show()
        logger.info(f"Precision-Recall Curve for {classifier_name} classification model.")
    
    return fig, ax
