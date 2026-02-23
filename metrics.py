import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_confusion(y_true, y_pred, labels):

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc(y_true, probs, n_classes):

    y_true_bin = np.eye(n_classes)[y_true]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:,i], probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")

    plt.legend()
    plt.title("ROC Curve")
    plt.show()