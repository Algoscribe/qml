import random
from utils import entropy
from noise_model import apply_all_noise

def simulate(attack="normal"):

    n = 200

    distance = random.uniform(5,50)
    alpha = 0.2
    transmittance = 10**(-(alpha*distance)/10)

    alice_bits  = [random.randint(0,1) for _ in range(n)]
    alice_bases = [random.randint(0,1) for _ in range(n)]
    bob_bases   = [random.randint(0,1) for _ in range(n)]

    sifted_alice=[]
    sifted_bob=[]

    for i in range(n):

        bit = alice_bits[i]
        if attack in ["rng","combined"]:
            bit = 1 if random.random()<0.8 else 0

        basis = alice_bases[i]
        photons = random.choice([1,1,1,2])

        if random.random()>transmittance:
            continue

        if attack in ["intercept","combined"]:
            if random.randint(0,1)!=basis:
                bit=random.randint(0,1)

        if attack in ["pns","combined"] and photons>1:
            continue

        if attack in ["trojan","combined"] and random.random()<0.1:
            bit=random.randint(0,1)

        if attack in ["blinding","combined"] and random.random()<0.4:
            continue

        if attack in ["wavelength","combined"] and random.random()<0.3:
            bit=1

        if basis==bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            
            sifted_bob.append(apply_all_noise(bit))

    if len(sifted_bob)==0:
        return 0,0,0,0,0

    errors=sum(a!=b for a,b in zip(sifted_alice,sifted_bob))
    qber=errors/len(sifted_alice)

    ones=sum(sifted_bob)
    zeros=len(sifted_bob)-ones
    bias=abs(ones-zeros)/len(sifted_bob)

    ent=entropy(sifted_bob)
    loss=1-transmittance

    return qber,len(sifted_alice),bias,ent,loss