# =========================================================
# CONFUSION MATRIX (STABLE VERSION)
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def show_confusion():

    print("\nGenerating Confusion Matrix...\n")

    # ---- Load dataset ----
    data = pd.read_csv("dataset.csv")

    X = data.drop("label", axis=1)
    y = data["label"]

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ---- Train model ----
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # ---- Confusion matrix ----
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    plt.show()

    print("Confusion matrix displayed.\n")