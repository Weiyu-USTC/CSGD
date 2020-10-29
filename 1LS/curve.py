import numpy as np
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


censor_comm = np.load('./result/D10_alpha0.2/'+ '0comm.npy')
censor_norm = np.load('./result/D10_alpha0.2/'+ '0x_star_norm.npy')


no_comm = np.load('./result/D10_alpha0.2/'+ '1comm.npy')
no_norm = np.load('./result/D10_alpha0.2/'+ '1x_star_norm.npy')

lag_comm = np.load('./result/D10_alpha0.2/'+ '2comm.npy')
lag_norm = np.load('./result/D10_alpha0.2/'+ '2x_star_norm.npy')

# localSGD_comm = np.load('./result/D10_alpha0.2/'+ '3comm.npy')
# localSGD_norm = np.load('./result/D10_alpha0.2/'+ '3x_star_norm.npy')

fontsize = 18
legsize = 18

plt.semilogy(censor_comm, censor_norm, '-', linewidth=3.0, label='CSGD')
plt.plot(no_comm, no_norm, 'o--', markevery = (6,10), markeredgewidth = 3.0, linewidth=3.0, label='SGD')
plt.plot(lag_comm, lag_norm, 'v-.', markevery = (1,10), markeredgewidth = 3.0, linewidth=3.0, label='LAG-S')
# plt.plot(localSGD_comm, localSGD_norm, '--', linewidth=3.0, label='Local SGD')
plt.ylim((1e-5,1e-1))
plt.xlim((0,500))
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('ls_comm.pdf', format='pdf', bbox_inches='tight')
plt.show()


plt.semilogy(np.arange(len(censor_norm)), censor_norm, '-', linewidth=3.0, label='CSGD')
plt.semilogy(np.arange(len(no_norm)),  no_norm, 'o--', markevery = (6,10), markeredgewidth = 3.0, linewidth=3.0, label='SGD')
plt.semilogy(np.arange(len(lag_norm)),  lag_norm, 'v-.', markevery = (1,10), markeredgewidth = 3.0, linewidth=3.0, label='LAG-S')
# plt.plot(np.arange(len(localSGD_norm)), localSGD_norm, '--', linewidth=3.0, label='Local SGD')
plt.ylim((1e-5,1e-1))
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
plt.savefig('ls_iter.pdf', format='pdf', bbox_inches='tight')
plt.show()