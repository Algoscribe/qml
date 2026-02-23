# =========================================================
# NOISE ROBUSTNESS TEST
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def add_noise(X,level):
    noise = np.random.normal(0,level,X.shape)
    return X + noise


def noise_experiment():

    data = pd.read_csv("dataset.csv")

    X = data.drop("label",axis=1).values
    y = data["label"].values

    levels = [0,0.05,0.1,0.2,0.3]
    accs = []

    for n in levels:

        Xn = add_noise(X,n)

        Xtr,Xte,ytr,yte = train_test_split(Xn,y,test_size=0.3)

        model = RandomForestClassifier()
        model.fit(Xtr,ytr)

        acc = model.score(Xte,yte)
        accs.append(acc)

        print("Noise",n,"Accuracy",round(acc,3))

    plt.plot(levels,accs,marker="o")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Robustness Test")
    plt.show()