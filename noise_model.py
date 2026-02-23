import random
import numpy as np

def apply_noise(bit, noise_level=0.05):
    if random.random() < noise_level:
        return 1-bit
    return bit


def detector_dark_counts(bit, prob=0.02):
    if random.random() < prob:
        return random.randint(0,1)
    return bit


def phase_noise(bit, prob=0.03):
    if random.random() < prob:
        return 1-bit
    return bit


def polarization_drift(bit, prob=0.04):
    if random.random() < prob:
        return random.randint(0,1)
    return bit


def apply_all_noise(bit):
    bit = apply_noise(bit)
    bit = detector_dark_counts(bit)
    bit = phase_noise(bit)
    bit = polarization_drift(bit)
    return bit