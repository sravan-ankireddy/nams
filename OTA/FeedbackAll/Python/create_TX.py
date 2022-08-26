import numpy as np
from commpy import QAMModem
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import kaiserord
import scipy.io as sio
import pickle


def filter_data(X, filter=True):

    # Compute the order and Kaiser parameter for the FIR filter.
    width = 0.1
    ripple_db = 60
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 0.5

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = signal.firwin(N, cutoff_hz, window=("kaiser", beta))

    if filter:
        X_f = signal.lfilter(taps, 1, X)[: len(X)]
        return X_f
    else:
        return X


def contellation(N1, N2, M):

    m = int(np.log2(M))
    QAM = QAMModem(M)

    Data = np.zeros((N1, N2)) + 1j * np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            bit = [int(x) for x in bin(np.random.randint(0, M))[2:]]
            l = len(bit)
            D = np.zeros(int(np.log2(M)), dtype=int)
            D[m - l :] = bit
            Data[i, j] = QAM.modulate(D)
    return Data


def generate_preamble(NFFT):

    cp = int(NFFT / 4)
    STS = pickle.load(open("STS.pkl", "rb"))
    STS = np.array(STS["S"]).reshape(
        NFFT,
    )
    X = np.fft.ifft(np.fft.fftshift(STS), NFFT)
    A = np.concatenate((X[NFFT - cp : NFFT], X))
    A = np.concatenate((A, A))
    A = np.concatenate((A, A[0:1]))

    # Windowed
    A[0] = A[0] * 0.5
    A[-1] = A[-1] * 0.5

    B1 = np.zeros((2 * (NFFT + cp) + 1, 2))
    B1[:, 0] = np.real(A)
    B1[:, 1] = np.imag(A)
    B1 = B1.flatten()

    cp = int(NFFT / 4)
    LTS = pickle.load(open("LTS.pkl", "rb"))
    LTS = np.array(LTS["L"]).reshape(
        NFFT,
    )
    X = np.fft.ifft(np.fft.fftshift(LTS), NFFT)
    A = np.concatenate((X[NFFT - cp : NFFT], X))
    A = np.concatenate((A, A))
    A = np.concatenate((A, A[0:1]))

    # Windowed
    A[0] = A[0] * 0.5
    A[-1] = A[-1] * 0.5

    B2 = np.zeros((2 * (NFFT + cp) + 1, 2))
    B2[:, 0] = np.real(A)
    B2[:, 1] = np.imag(A)
    B2 = B2.flatten()
    B1[-2:] = B1[-2:] + B2[:2]
    B = np.concatenate((B1, B2[2:]))

    return B


def OFDM_Symb(N, NFFT, Data):

    # This function arranges the data as [IQIQIQIQ...]
    cp = int(NFFT / 4)
    A = np.zeros((N, NFFT + cp)) + 1j * np.zeros((N, NFFT + cp))
    B = np.zeros((N * (NFFT + cp), 2))
    for i in range(N):
        X = np.zeros((NFFT)) + 1j * np.zeros((NFFT))
        for j in np.concatenate((np.arange(6, 31), np.arange(33, 57))):
            X[j] = Data[i, j]
        IFFT_Data = np.fft.ifft(np.fft.fftshift(X), NFFT)
        IFFT_Data = np.concatenate((IFFT_Data[NFFT - cp : NFFT], IFFT_Data))
        A[i, :] = filter_data(IFFT_Data, False)
    A = A.flatten()
    B[:, 0] = np.real(A)
    B[:, 1] = np.imag(A)
    B = B.flatten()
    return B


# # Parameters
# N = 100
# NFFT = 64
# M = 4

# # Short Training Sequence
# X1 = generate_preamble(NFFT)

# # Long Training Sequence

# # Data
# Data = contellation(N, NFFT, M)
# X2 = OFDM_Symb(N, NFFT, Data)


# X = np.concatenate((X1, X2))

# X.astype("float32").tofile("TX.bin")

STS = pickle.load(open("STS.pkl", "rb"))
print(STS)

# LTS_FD = [
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     1,
#     1,
#     -1,
#     -1,
#     1,
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     -1,
#     -1,
#     1,
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     1,
#     1,
#     1,
#     0,
#     1,
#     -1,
#     -1,
#     1,
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     -1,
#     -1,
#     -1,
#     -1,
#     -1,
#     1,
#     1,
#     -1,
#     -1,
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     1,
#     1,
#     1,
#     0,
#     0,
#     0,
#     0,
#     0,
# ]
# print(len(LTS_FD))
# LTS_FD = {"L": LTS_FD}
# pickle.dump(LTS_FD, open("LTS.pkl", "wb"))
