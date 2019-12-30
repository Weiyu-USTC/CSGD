# coding:utf-8
import numpy as np
import re
from sklearn.model_selection import StratifiedShuffleSplit

path = "./dataSet/covtype.libsvm.binary.scale"
set_size = 581012
n_feature = 54
def load_data(path):
	X = np.zeros((set_size, n_feature), dtype=np.float64)
	Y = []
	with open(path, 'r') as f:
		for (i, line) in enumerate(f):
			if i == 0:
				print line
			(label, data) = line.split(' ', 1)
			if label == '1':
				Y.append(1)
			else:
				Y.append(0)
			for pairs in data.strip().split(' '):
				match = re.search(r'(\S+):(\S+)', pairs)
				feature = int(match.group(1)) - 1 # 数据集从1开始
				value = float(match.group(2))
				X[i][feature] = value
	print "X shape:", X.shape
	print "Y:", len(Y)
	np.save("./dataSet/" + "x.npy", X)
	np.save("./dataSet/" + "y.npy", Y)
if __name__ == "__main__":
	# load_data(path)
	x = np.load("./dataSet/x.npy")
	y = np.load("./dataSet/y.npy")
	print "x shape:", x.shape
	print "y shape:", y.shape
	sss=StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
	for train, test in sss.split(x, y):
		x_train, x_test = x[train], x[test]
		y_train, y_test = y[train], y[test]
	# np.save("./dataSet/x_train.npy", x_train)
	# np.save("./dataSet/x_test.npy", x_test)
	# np.save("./dataSet/y_train.npy", y_train)
	# np.save("./dataSet/y_test.npy", y_test)
	print "x_train: ", x_train.shape
	print "x_test: ", x_test.shape
	print "y_train: ", y_train.shape
	print "y_test: ", y_test.shape
