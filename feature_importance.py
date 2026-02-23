import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def show_feature_importance():

    df = pd.read_csv("dataset.csv")

    X = df.drop("label",axis=1)
    y = df["label"]

    model = RandomForestClassifier()
    model.fit(X,y)

    importance = model.feature_importances_

    plt.bar(X.columns, importance)
    plt.title("Feature Importance")
    plt.show()