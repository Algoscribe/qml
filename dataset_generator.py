import csv
from simulator import simulate
from utils import ATTACKS

def generate_dataset(samples=25):

    with open("dataset.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["qber","keylen","bias","entropy","loss","label"])

        for atk in ATTACKS:
            for _ in range(samples):
                q,k,b,e,l=simulate(atk)
                w.writerow([q,k,b,e,l,atk])

    print("Dataset saved as dataset.csv")