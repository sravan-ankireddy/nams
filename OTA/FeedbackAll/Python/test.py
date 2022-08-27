import numpy as np

C = np.zeros((5, 2))
D = np.arange(5) + 1j * np.arange(5, 10)
C[:,0] = np.real(D)
C[:,1] = np.imag(D)
print(C)
print(C.flatten())
