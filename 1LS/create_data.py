import numpy as np

row = 1000
col = 10
A = np.random.normal(0, 1.0, size=(row, col))
y = np.random.uniform(-2.0, 2.0, size=(col, 1))
noise = np.random.normal(0, 0.1, size=(row, 1))
b = np.dot(A, y) + noise
print b.shape
np.save('./data/A.npy', A)
np.save('./data/y.npy', y)
np.save('./data/noise.npy', noise)
np.save('./data/b.npy', b)


