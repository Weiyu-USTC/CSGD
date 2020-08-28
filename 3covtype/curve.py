import numpy as np
# import random
import matplotlib.pyplot as plt

def file2list(filename):
    """read data from txt file"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    returnMat = []
    for line in arrayOLines:
        line = line.strip()
        returnMat.append(float(line))
    return returnMat


censor_comm = np.load('./result/covtype/alpha0.1_D10_step/'+ '0comm.npy')
censor_norm = np.load('./result/covtype/alpha0.1_D10_step/'+ '0loss.npy')


no_comm = np.load('./result/covtype/alpha0.1_D10_step/'+ '1comm.npy')
no_norm = np.load('./result/covtype/alpha0.1_D10_step/'+ '1loss.npy')

lag_comm = np.load('./result/covtype/alpha0.1_D10_step/'+ '2comm.npy')
lag_norm = np.load('./result/covtype/alpha0.1_D10_step/'+ '2loss.npy')

localSGD_comm = np.load('./result/covtype/LocalSGD-alpha-0.1-TIME-30/'+ 'comm.npy')
localSGD_norm = np.load('./result/covtype/LocalSGD-alpha-0.1-TIME-30/'+ 'loss.npy')

fontsize = 18
legsize = 22

plt.plot(censor_comm, censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(no_comm, no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(lag_comm, lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.plot(localSGD_comm, localSGD_norm, '-o', linewidth=2.0, label='Local SGD')
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('covtype_comm.pdf', format='pdf', bbox_inches='tight')
plt.show()

plt.plot(np.arange(25)*20+20, censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(np.arange(25)*20+20,  no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(np.arange(25)*20+20,  lag_norm , '-.', linewidth=2.0, label='LAG-S')
plt.plot(np.arange(25)*20+20, localSGD_norm, '-o', linewidth=2.0, label='Local SGD')
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('covtype_iter.pdf', format='pdf', bbox_inches='tight')
plt.show()