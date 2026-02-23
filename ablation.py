import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def ablation_test():

    df = pd.read_csv("dataset.csv")

    features = list(df.columns[:-1])

    for remove in features:

        X = df.drop(["label",remove],axis=1)
        y = df["label"]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

        model = RandomForestClassifier()
        model.fit(X_train,y_train)

        acc = accuracy_score(y_test,model.predict(X_test))

        print("Removed:",remove,"Accuracy:",round(acc,3))