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


censor_comm = np.load('./result/covtype/alpha0.1_D10_step/'+ '0comm.npy')
censor_norm = np.load('./result/covtype/alpha0.1_D10_step/'+ '0loss.npy')


no_comm = np.load('./result/covtype/alpha0.1_D10_step/'+ '1comm.npy')
no_norm = np.load('./result/covtype/alpha0.1_D10_step/'+ '1loss.npy')

lag_comm = np.load('./result/covtype/alpha0.1_D10_step/'+ '2comm.npy')
lag_norm = np.load('./result/covtype/alpha0.1_D10_step/'+ '2loss.npy')

fontsize = 18
legsize = 22

pdf = PdfPages('covtype_comm.pdf')
plt.plot(censor_comm, censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(no_comm, no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(lag_comm, lag_norm, '-.', linewidth=2.0, label='LAG-S')
plt.xlabel('communication cost', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
pdf.savefig()
pdf.close()
plt.show()

pdf = PdfPages('covtype_iter.pdf')
plt.plot(np.arange(25)*20+20, censor_norm, '--', linewidth=2.0, label='CSGD')
plt.plot(np.arange(25)*20+20,  no_norm, '-', linewidth=2.0, label='SGD')
plt.plot(np.arange(25)*20+20,  lag_norm , '-.', linewidth=2.0, label='LAG-S')
plt.xlabel('iteration index k', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=legsize)
pdf.savefig()
pdf.close()
plt.show()