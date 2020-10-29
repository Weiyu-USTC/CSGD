import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
# np.set_printoptions(threshold='nan')
np.random.seed(3)

# ====================================================
def log(*k, **kw):
    timeStamp = time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)


num_class = 10
num_feature = 28 * 28
num_train = 60000
num_test = 10000
num_machines = 10

num_iter = 10
eta1 = 1.5
alpha = 1e-4
max_batch = (num_train / num_machines) / 1

def cal_total_grad(X, Y, theta):

    """
    :param X: shape(num_train, num_feature + 1)
    :param Y: labels' one_hot array, shape(num_train, num_class)
    :param theta: shape (num_class, num_feature+1)
    :param weight_lambda: scalar
    :return: grad, shape(num_class, num_feature+1)
    """
    m = X.shape[0]
    # loss = 0.0
    t = np.exp(np.dot(theta, X.T)) #(num_classes, num_samples)
    # t = t - np.max(t, axis=0)
    t_sum = np.sum(t, axis=0)
    pro = t / t_sum
    total_grad = - np.dot((Y.T - pro), X) / m
    # add regularization term
    # weight_lambda = 0.001
    # total_grad = total_grad + weight_lambda * theta
    return total_grad


def cal_acc(test_x, test_y, theta):

    num = 0
    m = test_x.shape[0]
    for i in range(m):
        t1 = np.dot(theta, test_x[i])
        # t1 = t1 - np.max(t1, axis=0)
        pro = np.exp(t1) # un-normalized prob
        if np.argmax(pro) == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc


def cal_mean(grad_li):
    return sum(grad_li) / len(grad_li)

class Machine:
    def __init__(self, data_x, data_y, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(num_samples/num_machines, dimension)
        data_x : a numpy array has shape :num_samples/num_machines, dimension)
        data_y: a list of length 'num_samples/num_machine', the label of the data_x"""

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id

    def update(self, theta, batch_size):
        """Calculates gradient with a randomly selected sample, given the current theta
         Accepts theta, a np array with shape of (dimension,)
         Returns the calculated gradient"""
        m = self.data_x.shape[0]
        # print "m:", m
        # print batch_size
        if batch_size >= m:
            # batch_size = m
            grad_f = cal_total_grad(self.data_x, self.data_y, theta)
        else:
            begin = random.randint(0, m)
            idx = [i % m for i in range(begin, begin + batch_size)]
            grad_f = cal_total_grad(self.data_x[idx], self.data_y[idx], theta)
        # idx = [random.randint(0, m-1) for i in range(0, batch_size)]
        # grad_f = cal_total_grad(self.data_x[idx], self.data_y[idx], theta)
        return grad_f


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.theta = [] #list that stores each theta, grows by one iteration
        self.acc_li = []
        
        train_img = np.load('./data/mnist/train_img.npy')  # shape(60000, 784)
        train_lbl = np.load('./data/mnist/train_lbl.npy')  # shape(60000,)
        one_train_lbl = np.load('./data/mnist/one_train_lbl.npy')  # shape(60000, 10)
        test_img = np.load('./data/mnist/test_img.npy')  # shape(10000, 784)
        test_lbl = np.load('./data/mnist/test_lbl.npy')  # shape(10000,)

        bias_train = np.ones(num_train)
        train_img_bias = np.column_stack((train_img, bias_train))

        bias_test = np.ones(num_test)
        test_img_bias = np.column_stack((test_img, bias_test))

        self.test_img_bias = test_img_bias
        self.test_lbl = test_lbl
        self.train_img_bias = train_img_bias
        self.one_train_lbl = one_train_lbl
        self.train_lbl = train_lbl

        samples_per_machine = int(num_train / num_machines)

        self.machines = []
        #########  i.i.d case
        for i in range(num_machines):
            new_machine = Machine(train_img_bias[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  one_train_lbl[i * samples_per_machine:(i + 1) * samples_per_machine, :], i)
            self.machines.append(new_machine)

    def train(self, init_theta, alpha):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""
        
        # print "len pre_grad:", len(self.pre_grad)
        self.theta.append(init_theta)
        theta = init_theta
        batch_size = eta1
        grad_li = []
        for i,mac in enumerate(self.machines):
            grad_li.append(theta.copy())
        # there's a scaling, which should be equivalently put in the gradients.
        for i in range(num_iter):
            for j, mac in enumerate(self.machines):
                grad_li[j] = mac.update(theta, int(batch_size))
            mean_grad = cal_mean(grad_li)
            theta = theta - alpha * mean_grad
            self.theta.append(theta)
            err = 1 - cal_acc(self.test_img_bias, self.test_lbl, theta)
            self.acc_li.append(err)
            batch_size = min(max_batch, batch_size * eta1)
            # log("step:", i, " comm:", self.temp_comm[-1])
        log(err) # print("train end!")

def init():
    server = Parameter_server()
    return server

def main():
    np.random.seed(3)
    init_theta = np.zeros((num_class, num_feature + 1))
    server = init()
    server.train(init_theta, alpha)
    np.save('./init_theta.npy', server.theta[-1])

main()