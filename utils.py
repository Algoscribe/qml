import numpy as np

# =========================================================
# ALL SUPPORTED ATTACK TYPES
# =========================================================
ATTACKS = [
    "normal",
    "intercept",
    "pns",
    "trojan",
    "blinding",
    "rng",
    "wavelength",
    "combined"
]


# =========================================================
# ENTROPY FUNCTION
# =========================================================
def entropy(bits):

    if len(bits) == 0:
        return 0.0

    bits = np.array(bits)

    p1 = np.mean(bits)
    p0 = 1 - p1

    def h(p):
        return -p*np.log2(p) if p > 0 else 0

    return h(p1) + h(p0)


# =========================================================
# VALIDATION HELPER
# =========================================================
def is_valid_attack(name):
    return name.lower() in ATTACKS