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


censor_comm = np.load('./result/alpha0.1_D10/'+ '0comm.npy')
censor_norm = np.load('./result/alpha0.1_D10/'+ '0acc.npy')


no_comm = np.load('./result/alpha0.1_D10/'+ '1comm.npy')
no_norm = np.load('./result/alpha0.1_D10/'+ '1acc.npy')

lag_comm = np.load('./result/alpha0.1_D10/'+ '2comm.npy')
lag_norm = np.load('./result/alpha0.1_D10/'+ '2acc.npy')

localSGD_comm = np.load('./result/LocalSGD-alpha-0.01-TIME-30/'+ 'comm.npy')
localSGD_norm = np.load('./result/LocalSGD-alpha-0.01-TIME-30/'+ 'acc.npy')

fontsize = 18
legsize = 22


plt.semilogy(censor_comm, censor_norm, '--', subsy=[], linewidth=2.0, label='CSGD')
plt.plot(no_comm, no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(lag_comm, lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.plot(localSGD_comm, localSGD_norm, '-o', linewidth=2.0, label='Local SGD')
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.6,1.0])
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylim((0.07,1))
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('mnist_comm.pdf', format='pdf', bbox_inches='tight')
plt.show()

plt.semilogy(np.arange(85)*2, censor_norm, '--', subsy=[], linewidth=2.0, label='CSGD')
plt.semilogy(np.arange(85)*2, no_norm, '-', linewidth=2.0, label='SGD')
plt.semilogy(np.arange(85)*2, lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.semilogy(np.arange(85)*2, localSGD_norm, '-o', linewidth=2.0, label='Local SGD')
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.6,1.0])
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylim((0.07,1))
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('mnist_iter.pdf', format='pdf', bbox_inches='tight')
plt.show()