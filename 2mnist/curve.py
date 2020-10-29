import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def file2list(filename):
    """read data from txt file"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    returnMat = []
    for line in arrayOLines:
        line = line.strip()
        returnMat.append(float(line))
    return returnMat

s1 = 'alpha2e-5_D10/'

censor_comm = np.load('./result/' + s1 + '0comm.npy')
censor_norm = np.load('./result/' + s1 + '0acc.npy')


no_comm = np.load('./result/' + s1 + '1comm.npy')
no_norm = np.load('./result/' + s1 + '1acc.npy')

lag_comm = np.load('./result/' + s1 + '2comm.npy')
lag_norm = np.load('./result/' + s1 + '2acc.npy')

# localSGD_comm = np.load('./result/' + s1 + '3comm.npy')
# localSGD_norm = np.load('./result/' + s1 + '3acc.npy')

fontsize = 18
legsize = 18


plt.semilogy(censor_comm, censor_norm, '-', subs=[], linewidth=3.0, label='CSGD')
plt.plot(no_comm, no_norm, 'o--', markevery = (11,20), markeredgewidth = 3.0, linewidth=3.0, label='SGD')
plt.plot(lag_comm, lag_norm, 'v-.', markevery = (1,20), markeredgewidth = 3.0, linewidth=3.0, label='LAG-S')
# plt.plot(localSGD_comm, localSGD_norm, '--', linewidth=3.0, label='Local SGD')
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5])
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylim((0.1,0.5))
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('mnist_comm.pdf', format='pdf', bbox_inches='tight')
plt.show()

plt.semilogy(np.arange(len(censor_norm))*2, censor_norm, '-', subs=[], linewidth=3.0, label='CSGD')
plt.semilogy(np.arange(len(no_norm))*2, no_norm, 'o--', markevery = (11,20), markeredgewidth = 3.0, linewidth=3.0, label='SGD')
plt.semilogy(np.arange(len(lag_norm))*2, lag_norm, 'v-.', markevery = (1,20), markeredgewidth = 3.0, linewidth=3.0, label='LAG-S')
# plt.semilogy(np.arange(len(localSGD_norm))*2, localSGD_norm, '--', linewidth=3.0, label='Local SGD')
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5])
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylim((0.1,0.5))
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('mnist_iter.pdf', format='pdf', bbox_inches='tight')
plt.show()