import matplotlib.pyplot as plt
import pickle
# %matplotlib notebook

fig, axs = plt.subplots(1, 2)

plans = [
    ('CIFAR10_ResNet_SGD', 'SGD', '-'),
    ('CIFAR10_ResNet_LAG_S', 'LAG_S', 'o--'),
    ('CIFAR10_ResNet_CSGD', 'CSGD', 'v-.'),
]

for recordName, label, fmt in plans:
    with open('./cache/' + recordName, 'rb') as f:
        record = pickle.load(f)
        # 损失函数
        path = record['valAccPath']
        path = [1-p for p in path]

        com = record['communicationPath']
        axs[0].plot(com, path, fmt, label=label, 
            markevery = (11,20), markeredgewidth = 3.0, linewidth=3.0)

        assert len(path) == record['rounds'] + 1
        iterations = [record['displayInterval']*i for i in range(len(path))]
        axs[1].plot(iterations, path, fmt, label=label, 
            markevery = (11,20), markeredgewidth = 3.0, linewidth=3.0)

for ax in axs:
    # ax.legend(loc='lower left', bbox_to_anchor=(0,-0.9))
    ax.legend()
axs[0].set_xlim(left=-2000, right=60000)
axs[1].set_xlim(left=-2000, right=30000)
axs[0].set_xlabel('communication cost')
axs[1].set_xlabel('iteration')
axs[0].set_ylim(top=0.6)
axs[1].set_ylim(top=0.6)
# axs[0].set_xscale('log')
SCALE = 0.5
fig.set_size_inches(12*SCALE, 6*SCALE)
plt.savefig('./cifar10.pdf', format='pdf', bbox_inches='tight')
plt.show()
