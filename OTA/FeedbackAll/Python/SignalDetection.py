import numpy as np


def powerTrigger(signal):
    L = len(signal)
    count = 0
    P = 0
    for i in range(L):
        if signal[i] * np.ctranspose(signal[i]) > 10:
            if P == 1:
                count += 1
            P = 1

        else:
            P = 0
            count = 0

        if count > 20:
            return signal[i:]
    return 0

def shortPreambleDetection(signal):
    