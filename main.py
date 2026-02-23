# =========================================================
# QKD MAIN CONTROLLER (FINAL CLEAN VERSION)
# =========================================================

from simulator import simulate
from dataset_generator import generate_dataset
from classical_model import train_classical
from quantum_model import train_quantum
from qlstm_model import train_qlstm
from utils import ATTACKS
import pandas as pd


# =========================================================
# RUN SINGLE ATTACK1
# =========================================================
def run_attack():

    print("\nAvailable attacks:")
    print(", ".join(ATTACKS))

    atk = input("\nEnter attack name: ").lower().strip()

    if atk not in ATTACKS:
        print("Invalid attack name.")
        return

    q,k,b,e,l = simulate(atk)

    print("\n===== RESULT =====")
    print("Attack:",atk.upper())
    print("QBER:",round(q,3))
    print("Key Length:",k)
    print("Bias:",round(b,3))
    print("Entropy:",round(e,3))
    print("Loss:",round(l,3))


# =========================================================
# SHOW DATASET
# =========================================================
def show_dataset():

    try:
        df = pd.read_csv("dataset.csv")

        print("\nDATASET SAMPLE\n")
        print(df.head(10))

        print("\nAttack Distribution\n")
        print(df["label"].value_counts())

    except:
        print("Dataset not found. Generate first.")


# =========================================================
# SHOW ALL ATTACKS
# =========================================================
def show_all():

    print("\n========== ALL ATTACK RESULTS ==========\n")

    for atk in ATTACKS:

        q,k,b,e,l = simulate(atk)

        print(atk.upper())
        print("QBER:",round(q,3),
              "Key:",k,
              "Bias:",round(b,3),
              "Entropy:",round(e,3),
              "Loss:",round(l,3))
        print("-"*40)


# =========================================================
# MENU
# =========================================================
def menu():

    while True:

        print("\n================================")
        print(" QKD ATTACK DETECTOR SYSTEM")
        print("================================")
        print("1 → Generate dataset")
        print("2 → Show dataset")
        print("3 → Run single attack")
        print("4 → Run all attacks")
        print("5 → Train classical model")
        print("6 → Train quantum model")
        print("7 → Train QLSTM model")
        print("8 → Feature importance")
        print("9 → Hyperparameter search")
        print("10 → Ablation study")
        print("0 → Exit")

        cmd = input("\nEnter choice: ").strip()

        # -------------------------

        if cmd == "1":
            generate_dataset()

        elif cmd == "2":
            show_dataset()

        elif cmd == "3":
            run_attack()

        elif cmd == "4":
            show_all()

        elif cmd == "5":
            train_classical()

        elif cmd == "6":
            train_quantum()

        elif cmd == "7":
            train_qlstm()

        elif cmd == "8":
            from feature_importance import show_feature_importance
            show_feature_importance()

        elif cmd == "9":
            from hyperparameter_search import test_learning_rates
            from quantum_model import VQC
            test_learning_rates(VQC)

        elif cmd == "10":
            from ablation import ablation_test
            ablation_test()

        elif cmd=="11":
            from statistics import evaluate_all
            evaluate_all()

        elif cmd=="12":
            from confusion import show_confusion
            show_confusion()

        elif cmd=="13":
            from noise_test import noise_experiment
            noise_experiment()

        elif cmd=="14":
            from results_table import show_table
            show_table()

        elif cmd == "0":
            print("Exiting system.")
            break

        else:
            print("Invalid choice.")


# =========================================================
if __name__ == "__main__":
    menu()