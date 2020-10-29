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

scale = 10 # how many iterations between two consecutive loss computation
s1 = 'alpha1_D10_step/'

censor_comm = np.load('./result/' + s1 + '0comm.npy')
censor_norm = np.load('./result/' + s1 + '0loss.npy')


no_comm = np.load('./result/' + s1 + '1comm.npy')
no_norm = np.load('./result/' + s1 + '1loss.npy')

lag_comm = np.load('./result/' + s1 + '2comm.npy')
lag_norm = np.load('./result/' + s1 + '2loss.npy')

# localSGD_comm = np.load('./result/' + s1 + '3comm.npy')
# localSGD_norm = np.load('./result/' + s1 + '3loss.npy')

fontsize = 18
legsize = 18

plt.plot(censor_comm, censor_norm, '-', linewidth=3.0, label='CSGD')
plt.plot(no_comm, no_norm, 'v--', markevery = (6,10), markeredgewidth = 3.0, linewidth=3.0, label='SGD')
plt.plot(lag_comm, lag_norm, 'o-.', markevery = (1,10), markeredgewidth = 3.0, linewidth=3.0, label='LAG-S')
# plt.plot(localSGD_comm, localSGD_norm, '--', linewidth=3.0, label='Local SGD')
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('covtype_comm.pdf', format='pdf', bbox_inches='tight')
plt.show()

plt.plot(np.arange(len(censor_norm))*scale, censor_norm, '-', linewidth=3.0, label='CSGD')
plt.plot(np.arange(len(no_norm))*scale+scale,  no_norm, 'v--', markevery = (6,10), markeredgewidth = 3.0, linewidth=3.0, label='SGD')
plt.plot(np.arange(len(lag_norm))*scale+scale,  lag_norm , 'o-.', markevery = (1,10), markeredgewidth = 3.0, linewidth=3.0, label='LAG-S')
# plt.plot(np.arange(len(localSGD_norm))*scale+scale, localSGD_norm, '--', linewidth=3.0, label='Local SGD')
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('covtype_iter.pdf', format='pdf', bbox_inches='tight')
plt.show()