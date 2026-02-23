import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split

# =====================================================
# QLSTM CELL (REAL QUANTUM)
# =====================================================

class QLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # qubits must match feature size
        self.n_qubits = input_size + hidden_size

        # quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # trainable quantum weights
        self.weights = nn.Parameter(torch.randn(2, self.n_qubits, 3))

        # classical projection
        self.fc = nn.Linear(self.n_qubits, hidden_size)

        # quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit


    def forward(self, x, h):

        combined = torch.cat([x, h], dim=1)

        outputs = []

        for sample in combined:
            q = self.circuit(sample, self.weights)
            q = torch.stack(q)
            outputs.append(q)

        outputs = torch.stack(outputs).float()

        h_new = torch.tanh(self.fc(outputs))

        return h_new


# =====================================================
# QLSTM NETWORK
# =====================================================

class QLSTM(nn.Module):

    def __init__(self, input_size=5, hidden_size=6):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = QLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 8)


    def forward(self, x):

        batch = x.size(0)
        h = torch.zeros(batch, self.hidden_size)

        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)

        return self.fc(h)


# =====================================================
# SEQUENCE BUILDER
# =====================================================

def create_seq(X, y, L=4):

    xs, ys = [], []

    for i in range(len(X)-L):
        xs.append(X[i:i+L])
        ys.append(y[i+L])

    return np.array(xs), np.array(ys)


# =====================================================
# TRAIN FUNCTION
# =====================================================

def train_qlstm():

    print("\nTRAINING REAL QLSTM MODEL...\n")

    data = pd.read_csv("dataset.csv")

    X = data.drop("label", axis=1).values
    y = data["label"].astype("category").cat.codes.values

    Xs, ys = create_seq(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.25, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    model = QLSTM()

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # ================= TRAIN =================
    for e in range(10):

        pred = model(X_train)
        loss = loss_fn(pred, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = (pred.argmax(1) == y_train).float().mean()

        print(f"Epoch {e+1}/10 | Loss={loss.item():.4f} | Train Acc={acc:.4f}")

    # ================= TEST =================
    with torch.no_grad():
        pred_test = model(X_test)
        acc_test = (pred_test.argmax(1) == y_test).float().mean()
    

    print("\nFINAL TEST ACCURACY:", round(acc_test.item(), 4))
    print("\nREAL QLSTM TRAINING COMPLETE")

    return acc_test.item()