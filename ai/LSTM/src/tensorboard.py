"""Per-epoch binary classification charts for TensorBoard."""
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def log_predictions(writer, split, labels, scores, predictions, epoch):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure = Figure(figsize=(5, 4), tight_layout=True)
    ax = figure.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Alert", "Drowsy"],
           yticklabels=["Alert", "Drowsy"], xlabel="Predicted", ylabel="True",
           title=f"{split}: confusion matrix")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    writer.add_figure(f"{split}/confusion_matrix", figure, epoch)
    for label, name in enumerate(("alert", "drowsy")):
        values = scores[labels == label]
        writer.add_scalar(f"{split}/support_{name}", len(values), epoch)
        if len(values):
            writer.add_histogram(f"{split}/prob_drowsy_given_{name}", values, epoch)
    # ROC/AUC is undefined for a split containing only one class.
    if len(np.unique(labels)) == 2:
        writer.add_pr_curve(f"{split}/precision_recall", labels, scores, epoch)
        writer.add_scalar(f"{split}/roc_auc", roc_auc_score(labels, scores), epoch)
        writer.add_scalar(f"{split}/average_precision", average_precision_score(labels, scores), epoch)
        fpr, tpr, _ = roc_curve(labels, scores)
        figure = Figure(figsize=(5, 4), tight_layout=True)
        ax = figure.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"{split}: ROC", xlim=(0, 1), ylim=(0, 1))
        writer.add_figure(f"{split}/roc", figure, epoch)
