import numpy as np
import random
import matplotlib.pyplot as plt
import os
import matplotlib.colors

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# s1 = 'b2000/alpha0.4-skip40-sqrt-3-160-0.07'
# b2000 - 1
# # s1 = 'alpha0.4-D6-skip80-sqrt-3'
s1 = 'alpha0.1_D10_step/'
path = "./result/covtype/"
# path = r'G:\python_code\censorSgd\covtype2\covtype\result\covtype'

# path = os.path.join(path, s1)


file = 'comm-event.npy'

# comm = np.load(os.path.join(path, file))
comm = np.load(path + s1 + file)
comm = np.delete(comm, 0, axis=1)

print comm.shape

comm = comm.astype(np.float32)
comm[comm == 0.0] = np.nan
mark = np.ma.masked_where(np.isnan(comm), comm)


# t = comm[5].reshape(1, 500)
# t = t.astype(np.float32)
# print t.dtype
# print t
# t[t == 0.0] = np.nan
# m = np.ma.masked_where(np.isnan(t), t)

worker_li = [5,6,7,8,9]
# worker_li = [0, 1, 2, 3, 4]
size = len(worker_li)
plt.figure(figsize=(10, 6))
fontsize = 20
legsize = 22
cmap = matplotlib.colors.ListedColormap(['blue'])

for i in range(size):
    plt.subplot(size,1,i + 1)
    plt.pcolor(mark[worker_li[i]].reshape(1, 500), linewidths=2.0, cmap=cmap)
    if i != size - 1:
        plt.xticks([])
    else:
        plt.xticks(fontsize=16)
    yticks = [0, 1]
    plt.yticks(yticks, fontsize=16)
# plt.title('worker 6', fontsize=fontsize)
    s1 = 'WK ' + str(worker_li[i] + 1)
    plt.ylabel(s1, fontsize=fontsize)
plt.xlabel('Iteration index k', fontsize=fontsize)

plt.show()


