# =========================================================
# FINAL RESULTS TABLE
# =========================================================

def show_table():

    print("\nFINAL MODEL COMPARISON\n")

    print("Model\t\tAccuracy")
    print("---------------------------")

    from classical_model import train_classical
    from quantum_model import train_quantum
    from qlstm_model import train_qlstm

    print("Classical\t",train_classical(return_acc=True))
    print("Quantum\t\t",train_quantum(return_acc=True))
    print("QLSTM\t\t",train_qlstm(return_acc=True))