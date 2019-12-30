import numpy as np
# import random
from matplotlib.backends.backend_pdf import PdfPages
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


censor_comm = np.load('./result/D10_alpha0.02/'+ '0comm.npy')
censor_norm = np.load('./result/D10_alpha0.02/'+ '0x_star_norm.npy')


no_comm = np.load('./result/D10_alpha0.02/'+ '1comm.npy')
no_norm = np.load('./result/D10_alpha0.02/'+ '1x_star_norm.npy')

lag_comm = np.load('./result/D10_alpha0.02/'+ '2comm.npy')
lag_norm = np.load('./result/D10_alpha0.02/'+ '2x_star_norm.npy')

fontsize = 18
legsize = 22

pdf = PdfPages('ls_comm.pdf')
plt.semilogy(censor_comm, censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(no_comm, no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(lag_comm, lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.ylim((1e-6,10))
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
pdf.savefig()
pdf.close()
plt.show()


pdf = PdfPages('ls_iter.pdf')
plt.semilogy(np.arange(170), censor_norm, '--', linewidth=2.0, label='CSGD')
plt.semilogy(np.arange(170),  no_norm, '-', linewidth=2.0, label='SGD')
plt.semilogy(np.arange(170),  lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.ylim((1e-6,10))
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
pdf.savefig()
pdf.close()
plt.show()