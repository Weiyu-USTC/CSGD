import numpy as np

# row = 1000
col = 10
# A = np.random.normal(0, 1.0, size=(row, col))
y = np.random.uniform(-2.0, 2.0, size=(col, 1))
# y[0:5, :] = np.ones((5,1))
# noise = np.random.normal(0, 0.1, size=(row, 1))
# b = np.dot(A, y) + noise
# print (b.shape)
# np.save('./data2/A.npy', A)
np.save('./data/y.npy', y)
# np.save('./data2/noise.npy', noise)
# np.save('./data2/b.npy', b)