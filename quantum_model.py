# =========================================================
# QUANTUM MODEL (FIXED + STABLE VERSION)
# =========================================================

import torch
import torch.nn as nn
import pandas as pd
import pennylane as qml


# =========================
# Quantum Device
# =========================
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)


# =========================
# Quantum Circuit
# =========================
@qml.qnode(dev, interface="torch")
def circuit(x, w):

    # Encode classical features into qubits
    qml.AngleEmbedding(x, wires=range(n_qubits))

    # Trainable quantum layers
    qml.StronglyEntanglingLayers(w, wires=range(n_qubits))

    # Measure expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# =========================
# VQC MODEL
# =========================
class VQC(nn.Module):

    def __init__(self):
        super().__init__()

        # quantum weights
        self.w = nn.Parameter(torch.randn(2, n_qubits, 3))

        # classical classifier
        self.fc = nn.Linear(n_qubits, 8)


    def forward(self, x):

        outputs = []

        for sample in x:
            q_out = circuit(sample, self.w)   # run quantum circuit
            q_out = torch.stack(q_out)        # list → tensor
            outputs.append(q_out)

        outputs = torch.stack(outputs).float()

        return self.fc(outputs)


# =========================
# TRAIN FUNCTION
# =========================
def train_quantum():

    print("\nTRAINING QUANTUM MODEL...\n")

    # ---- Load dataset ----
    data = pd.read_csv("dataset.csv")

    # features
    X = torch.tensor(
        data.drop("label", axis=1).values,
        dtype=torch.float32
    )

    # labels (MUST be long dtype)
    y = torch.tensor(
        data["label"].astype("category").cat.codes.values,
        dtype=torch.long
    )


    # ---- Model ----
    model = VQC()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()


    # ---- Training ----
    epochs = 5

    for epoch in range(epochs):

        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (pred.argmax(1) == y).float().mean()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss = {loss.item():.4f} | "
            f"Accuracy = {acc.item():.4f}"
        )

    print("\nQuantum training finished.\n")

    return acc.item()