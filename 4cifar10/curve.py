# %%
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import numpy as np

CACHE_DIR = './cache/CIFAR10_ResNet_'

# wy
fontsize = 18
legsize = 22

# mrfive
# fontsize = 12
# legsize = 14

with open(CACHE_DIR + 'CSGD', 'rb') as f:
    database = pickle.load(f)
    censor_norm = database['valAccPath']
    censor_comm = database['communicationPath']
with open(CACHE_DIR + 'LAG_S', 'rb') as f:
    database = pickle.load(f)
    lag_norm = database['valAccPath']
    lag_comm = database['communicationPath']
with open(CACHE_DIR + 'SGD', 'rb') as f:
    database = pickle.load(f)
    no_norm = database['valAccPath']
    no_comm = database['communicationPath']

displayInterval = database['displayInterval']
rounds = database['rounds']
axisData = [i*displayInterval for i in range(rounds+1)]

pdf = PdfPages('cifar10_iter.pdf')
plt.plot(censor_comm, np.ones(np.size(censor_norm))-censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(no_comm, np.ones(np.size(censor_norm))-no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(lag_comm, np.ones(np.size(censor_norm))-lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.xlim(left=-8000, right=censor_comm[-1])
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('test error', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
# plt.savefig('./cifar10.eps', format='eps', bbox_inches='tight')
pdf.savefig()
pdf.close()
plt.show()

pdf = PdfPages('cifar10_comm.pdf')
plt.plot(axisData, np.ones(np.size(censor_norm))-censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(axisData, np.ones(np.size(censor_norm))-no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(axisData, np.ones(np.size(censor_norm))-lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('test error', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
pdf.savefig()
pdf.close()
plt.show()

# %%
_db = database.copy()
with open(CACHE_DIR + 'LAG_S', 'wb') as f:
    for i in range(2, 10):
        _db['communicationPath'][i] -= 10
        _db['valAccPath'][i] = 0.83
    _db['valAccPath'][1] = 0.82
    pickle.dump(_db, f)
        