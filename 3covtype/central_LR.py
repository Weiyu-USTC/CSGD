# coding:utf-8
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import scipy.sparse
import scipy.sparse.linalg
import sys
import time
from sklearn import metrics

class Arguments():
	def __init__(self):
		self.seed = 3
		self.batch_size = 128
		self.lr = 0.2
		self.lbda = 0.001
		self.epochs = 1
		self.epsilon = 1e-7
		self.num_samples = 464809
		self.features = 54
		self.scale = 50

args = Arguments()

def sigmoid(x):
	
	return np.exp(np.fmin(x, 0)) / (1.0 + np.exp(-np.abs(x)))

def cal_grad(w, x, y, lbda):

	# x: shape(N, feature)
	# y: shape(N, 1)
	# w: shape(feature, 1)

	w_x_mul = np.dot(x, w)
	# h_w = 1.0 / (1 + np.exp(-w_x_mul))
	h_w = sigmoid(w_x_mul)
	d = y - h_w
	grad = np.dot(x.T, d) / x.shape[0] + lbda * w
	loss = -sum(y*np.log(h_w + args.epsilon) + (1-y)*np.log(1-h_w + args.epsilon)) / x.shape[0] + lbda * sum(w ** 2) / 2.0
	return grad, loss

def cal_auc_acc(w, x_test, y_test):

	w_x_mul = np.dot(x_test, w)
	# probaOfPositive = 1.0 / (1.0 + np.exp(-w_x_mul))
	probaOfPositive = sigmoid(w_x_mul)
	probaOfNegative = 1.0 - probaOfPositive
	proba = np.hstack((probaOfNegative, probaOfPositive))
	y_pred = np.argmax(proba, axis=1)
	acc = sum(y_pred == y_test) / float(y_test.shape[0])
	auc = metrics.roc_auc_score(y_test, probaOfPositive)
	confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
	return acc, auc, confusionMatrix

def glorot_normal(fan_in, fan_out):
	stddev = np.sqrt(2 / (fan_in + fan_out))
	return np.random.normal(0, stddev, (fan_in, fan_out))

def plot_curve(path):

	# path = "./result/cLR/"
	# acc_li = np.load(path + "acc.npy")
	# auc_li = np.load(path + "auc.npy")
	loss_li = np.load(path + "loss.npy")
	grad_norm = np.load(path + "grad_norm.npy")

	# plt.figure()
	# plt.plot(np.arange(len(acc_li)), acc_li)
	# plt.xlabel("epoch")
	# plt.ylabel("Accuracy")
	# plt.savefig(path + "acc.png")
	# plt.show()

	# plt.plot(np.arange(len(auc_li)), auc_li)
	# plt.xlabel("epoch")
	# plt.ylabel("Auc")
	# plt.savefig(path + "auc.png")
	# plt.show()

	plt.plot(np.arange(len(loss_li)), loss_li)
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.savefig(path + "loss.png")
	plt.show()

	plt.semilogy(np.arange(len(grad_norm)), grad_norm)
	plt.xlabel("epoch")
	plt.ylabel("grad_norm")
	plt.savefig(path + "grad.png")
	plt.show()


def train(x_train, y_train, x_test, y_tesy, lbda):

	n_samples, features = x_train.shape[0], x_train.shape[1]

	w = np.zeros((features, 1))
	# w = glorot_normal(features, 1)
	n_batchs = int(n_samples / args.batch_size)

	rng = np.random.RandomState(args.seed)
	indexes = np.arange(n_samples)
	rng.shuffle(indexes)

	acc_li = []
	auc_li = []
	grad_norm = []
	loss_li = []

	init_grad, init_loss = cal_grad(w, x_train, y_train, lbda)
	print("initial loss: {}, init grad norm: {}".format(init_loss, sum(init_grad ** 2)))
	# init_acc, init_auc, init_cf = cal_auc_acc(w, x_test, y_test)
	# print("init acc: {}, init_auc: {}, init confusion matrix: {}".format(init_acc, init_auc, init_cf))

	# acc_li.append(init_acc)
	# auc_li.append(init_auc)
	grad_norm.append(sum(init_grad ** 2))
	loss_li.append(init_loss)


	step = 0
	base = args.lr
	for e in range(args.epochs):
		lssum = 0
		for i_batch in range(n_batchs):
			cur_indexes = indexes[i_batch*args.batch_size:min((i_batch+1)*args.batch_size, n_samples)]
			x_batch = x_train[cur_indexes]
			y_batch = y_train[cur_indexes]
			grad, loss = cal_grad(w, x_batch, y_batch, args.lbda)
			w = w + args.lr * grad
			step += 1
			lssum += loss
			# print("epoch: {}. step:{}. loss: {}. grad: {}".format(e, step, loss, sum(grad ** 2)))
			if step % args.scale == 0:
				# print("epoch: {}. step:{}. loss: {}. grad: {}".format(e, step, loss, sum(grad ** 2)))
				# acc, auc, cf = cal_auc_acc(w, x_test, y_test)
				total_grad, total_loss = cal_grad(w, x_train, y_train, args.lbda)
				print("epoch: {}. step:{}. loss: {}. grad: {}".format(e, step, total_loss, sum(total_grad ** 2)))
				# acc_li.append(acc)
				# auc_li.append(auc)
				grad_norm.append(sum(total_grad ** 2))
				loss_li.append(loss)
				# print("test epoch: {}. step: {}. acc: {}, auc: {}".format(e, step, acc, auc))
	# print("confusion matrix:")
	# print(cf)

	path = "./result/covtype/cLR/"
	# np.save(path + "acc.npy", acc_li)
	# np.save(path + "auc.npy", auc_li)
	np.save(path + "grad_norm.npy", grad_norm)
	np.save(path + "loss.npy", loss_li)

if __name__ == "__main__":
	# set_size = 581012
	# features = 54
	# x = np.load("./dataSet/x.npy")
	# y = np.load("./dataSet/y.npy")
	path_data = "./data/covtype/"
	path_result = "./result/covtype/cLR/"
	# x_train = np.load(path_data + "x_train.npy")
	# x_test = np.load(path_data + "x_test.npy")
	# y_train = np.load(path_data + "y_train.npy").reshape(-1,1)
	# y_test = np.load(path_data + "y_test.npy")
	x = np.load(path_data + 'x.npy')
	y = np.load(path_data + 'y.npy').reshape(-1, 1)
	train(x, y, x, y, args.lbda)
	plot_curve(path_result)
	# dict = {}
	# for e in y_test:
	# 	if e not in dict:
	# 		dict[e] = 0
	# 	else:
	# 		dict[e] += 1
	# print dict
	# train(x_train, y_train, x_test, y_test, args.lbda)
	# plot_curve(path_result)
	# w = np.zeros((54, 1))
	# grad, loss = cal_grad(w, x_train, y_train, args.lbda)
	# print loss
	# print x_train[:128,:].shape
	# print y_train[:128].shape
	# print loss.shape
	# print grad.shape
