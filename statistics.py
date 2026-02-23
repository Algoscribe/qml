# =========================================================
# STATISTICAL EVALUATION
# =========================================================

import numpy as np
from classical_model import train_classical
from quantum_model import train_quantum
from qlstm_model import train_qlstm


def run_multiple(model_func, runs=5):

    scores = []

    for i in range(runs):
        acc = model_func(return_acc=True)
        scores.append(acc)

    scores = np.array(scores)

    print("\nResults over",runs,"runs")
    print("Mean Accuracy:", round(scores.mean(),4))
    print("Std Deviation:", round(scores.std(),4))


def evaluate_all():

    print("\n===== STATISTICAL EVALUATION =====")

    print("\nClassical Model")
    run_multiple(train_classical)

    print("\nQuantum Model")
    run_multiple(train_quantum)

    print("\nQLSTM Model")
    run_multiple(train_qlstm)