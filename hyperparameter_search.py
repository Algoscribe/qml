import torch
import pandas as pd
import numpy as np


def test_learning_rates(ModelClass):

    print("\nHYPERPARAMETER SEARCH (Learning Rate)\n")

    data = pd.read_csv("dataset.csv")

    X = torch.tensor(
        data.drop("label",axis=1).values,
        dtype=torch.float32
    )

    # ⭐ FIX IS HERE
    y = torch.tensor(
        data["label"].astype("category").cat.codes.values,
        dtype=torch.long
    )

    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    results = []

    for lr in learning_rates:

        model = ModelClass()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(3):
            pred = model(X)
            loss = loss_fn(pred,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = (pred.argmax(1)==y).float().mean().item()

        print(f"LR={lr} → Acc={acc:.3f}")

        results.append((lr,acc))

    best = max(results,key=lambda x:x[1])

    print("\nBEST LR:",best[0],"Accuracy:",round(best[1],3))