import numpy as np
import matplotlib.pyplot as plt

X = np.fromfile("TX.bin", dtype="float32")
X = X[0:2:] + 1j * X[1:2:]

Y = np.fromfile("RX.bin", dtype="float32")
Y = Y[0:2:] + 1j * Y[1:2:]

Z = np.correlate(X, Y, "same")


plt.plot(abs(Z))
plt.show()