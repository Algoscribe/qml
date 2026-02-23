# =========================================================
# CLASSICAL MODEL
# =========================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_classical(return_acc=False):

    print("\nCLASSICAL RESULTS")

    data = pd.read_csv("dataset.csv")

    X = data.drop("label",axis=1)
    y = data["label"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test,preds)

    print("Accuracy:",acc)
    print(classification_report(y_test,preds))

    if return_acc:
        return acc