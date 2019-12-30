import numpy as np
import random
# import matplotlib.pyplot as plt
import time
from math import log
# np.set_printoptions(threshold='nan')
np.random.seed(3)

num_class = 10
num_feature = 28 * 28
num_train = 60000
num_test = 10000
num_machines = 10

num_iter = 170
def cal_total_grad(X, Y, theta):

    """
    :param X: shape(num_samples, features + 1)
    :param Y: labels' one_hot array, shape(num_samples, num_classes)
    :param theta: shape (num_classes, feature+1)
    :param weight_lambda: scalar
    :return: grad, shape(num_classes, feature+1)
    """
    m = X.shape[0]
    # loss = 0.0
    t = np.dot(theta, X.T) #(num_classes, num_samples)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / m
    # add regularization term
    weight_lambda = 0.00#0.001
    total_grad = total_grad + weight_lambda * theta
    # loss = -np.sum(Y.T * np.log(pro)) / m + weight_lambda / 2 * np.sum(theta ** 2)
    return total_grad


def cal_loss(X, Y, theta, weight_lambda):

    m = X.shape[0]
    t1 = np.dot(theta, X.T)
    t1 = t1 - np.max(t1, axis=0)
    t = np.exp(t1)
    tmp = t / np.sum(t, axis=0)
    loss = -np.sum(Y.T * np.log(tmp)) / m + weight_lambda * np.sum(theta ** 2) / 2
    return loss


def cal_acc(test_x, test_y, theta):

    num = 0
    m = test_x.shape[0]
    for i in range(m):
        t1 = np.dot(theta, test_x[i])
        t1 = t1 - np.max(t1, axis=0)
        pro = np.exp(t1)
        index = np.argmax(pro)
        if index == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc


def cal_mean(grad_li):
    m = len(grad_li)
    grad = np.zeros_like(grad_li[0])
    grad = sum(grad_li)
    # grad = grad / m
    return grad


def cal_max_norm_grad(theta):
    if np.all(theta == 0):
        return theta
    tmp = np.abs(theta)
    re = np.where(tmp == np.max(tmp))
    row = re[0][0]
    col = re[1][0]
    max_val = tmp[row, col]
    n = len(re[0])
    theta[tmp != np.max(tmp)] = 0
    theta[theta == -max_val] = -1.0 / n
    theta[theta == max_val] = 1.0 / n
    return theta


def cal_var(theta):
    mean_theta = np.mean(theta, axis=0)
    mean_arr = np.tile(mean_theta, (theta.shape[0], 1))
    tmp = theta - mean_arr
    var = np.trace(np.dot(tmp, tmp.T))
    return var


def huber_loss_grad(e, d):

    t = (np.abs(e) <= d) * e
    e[np.abs(e) <= d] = 0
    grad = t + d * np.sign(e)
    return grad


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
            batch_size = m
        id = random.randint(0, m - batch_size)
        grad_f = cal_total_grad(self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], theta)
        return grad_f


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.old_theta = [] #list that stores each theta, grows by one iteration
        self.acc_li = []
        self.grad_li = []
        self.grad_norm = []
        self.acc_li = []
        self.loss_li = []
        self.comm_cost = []
        self.comm_cost.append(0)
        
        self.temp_comm = []
        self.temp_comm.append(0)
        self.comm_event = [[0] for _ in range(num_machines)]
        self.min_acc = []
        
        self.pre_grad = [np.zeros((num_class, num_feature + 1)) for _ in range(num_machines)]

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

        samples_per_machine = num_train / num_machines
        self.machines = []
        #########  i.i.d case
        for i in range(num_machines):
            new_machine = Machine(train_img_bias[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  one_train_lbl[i * samples_per_machine:(i + 1) * samples_per_machine, :], i)
            self.machines.append(new_machine)


    def broadcast(self, theta, old_theta, pre_grad, alpha, batch_size, control_size, D, step, censor_type):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(dimension,)
        Return a list of length 'num_machines' containing the updated theta of each machine"""

        new_grad_li = []
        comm = self.temp_comm[-1]
        if censor_type ==1:
            tao = 0.0   # SGD
        else:
            if len(old_theta) < D:
                if censor_type == 0:
                    tao = control_size/(num_machines**2)
                else: 
                    tao=0
            else:
                temp_diff = []
                for j in range(1, len(old_theta)):
                    temp_diff.append(np.sum((old_theta[j] - old_theta[j - 1]) ** 2))
                if censor_type == 0:
                    tao = (sum(temp_diff) / (alpha**2*60 *D) + control_size)/(num_machines**2)
                else:
                    tao = sum(temp_diff) / (alpha**2*60 *D*num_machines**2)
                #print "step:", step, "temp_diff:", temp_diff
        print("step:", step, "tao:", tao)


        batch_size = int(batch_size)
        for i, mac in enumerate(self.machines):
            cur_grad = mac.update(theta, batch_size)
            grad_diff = np.sum((cur_grad - pre_grad[i]) ** 2)
            # print(grad_diff)
            if grad_diff > tao:
                # print "step:", step, "mac:", i
                new_grad_li.append(cur_grad)
                pre_grad[i] = cur_grad
                comm += 1
                self.comm_event[i].append(1)
                print "step:", step, "mac:", i
            else:
                new_grad_li.append(pre_grad[i])
                self.comm_event[i].append(0)
        if step % 2 == 0:
            self.comm_cost.append(comm)
        self.temp_comm.append(comm)
        return new_grad_li
            

    def train(self, init_theta, alpha, D):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""
        
        # print "len pre_grad:", len(self.pre_grad)
        self.old_theta.append(init_theta)
        theta = init_theta
        eta1=1.05
        sigma0=15
        eta2=0.96
        batch_size=eta1
        control_size=sigma0*eta2*6000**2 
        # there's a scaling, which should be equivalently put in the gradients.
        for i in range(num_iter):
            rec_grad_li = self.broadcast(theta, self.old_theta, self.pre_grad, alpha, batch_size, control_size,  D, i, 0)
            mean_grad = cal_mean(rec_grad_li)
            theta = theta - alpha * mean_grad
            if len(self.old_theta) < D:
                self.old_theta.append(theta)
            else:
                self.old_theta.pop(0)
                self.old_theta.append(theta)
            if (i + 1) % 2 == 0:
                acc = 1-(cal_acc(self.test_img_bias, self.test_lbl, theta))
                self.acc_li.append(acc)
                #print "step:", i, " acc:", acc

            #self.x_li.append(x)
            #self.x_star_norm.append(np.linalg.norm(x - x_star))
            batch_size=batch_size*eta1
            control_size=control_size*eta2
            #print "step:", i, "grad_norm:", np.linalg.norm(mean_grad)
            #print "step:", i, "x_star_norm:", self.x_star_norm[-1]
        print("comm cost:", self.temp_comm[-1])
        # self.min_acc=self.acc_li
        # for i in range(len(self.acc_li)):
        #     if i == 0:
        #         self.min_acc[0]=self.acc_li[0]
        #     else:
        #         self.min_acc[i]=min(self.min_acc[i-1],self.acc_li[i])
        print("train end!")

    def plot_curve(self):
        """plot the loss curve and the acc curve
        save the learned theta to a numpy array and a txt file"""

        s1 = 'alpha0.1_D10'
        np.save('./result/' + s1 + '/0acc.npy', self.acc_li)

        plt.plot(np.arange(len(self.acc_li)) * 2, self.acc_li)
        plt.xlabel('iter')
        plt.ylabel('error')#plt.ylabel('min(error)')#
        # plt.title(s1)
        plt.savefig('./result/' + s1 + '/0acc.png')
        plt.show()
        
        self.comm_cost.pop(0)
        #print len(self.comm_cost)
        #print len(self.acc_li)
        np.save('./result/' + s1 + '/0comm.npy', self.comm_cost)
        #np.save('./result/' + s1 + '/comm-event.npy', self.comm_event)

        plt.plot(self.comm_cost, self.acc_li)
        plt.xlabel('communication cost')
        plt.ylabel('error')#plt.ylabel('min(error)')#
        # plt.title(s1)
        plt.savefig('./result/' + s1 + '/0comm.png')
        plt.show()

        # plt.semilogy(np.arange(num_iter), self.grad_norm)
        # plt.xlabel('iter')
        # plt.ylabel('log||grad||')
        # # plt.title(s1)
        # plt.savefig('./result/RSGD/no_fault/same_digit/' + s1 + '/grad_norm.png')
        # plt.show()

        # plt.semilogy(np.arange(num_iter), self.var_li)
        # plt.xlabel('iter')
        # plt.ylabel('log||var||')
        # plt.savefig('./result/RSGD/fault/same_attack/q8/' + s1 + '/var.png')
        # plt.show()


def init():
    server = Parameter_server()
    return server


def main():
    np.random.seed(3)
    server = init()
    init_theta = np.zeros((num_class, num_feature + 1))
    alpha = 0.01
    D = 10
    server.train(init_theta, alpha, D)
    server.plot_curve()


main()
# import cProfile
# import pstats
# s1 = 'lam0.07_wei0.01_alpha0.001_sqrt_time(test4)'
# cProfile.run("main()", filename="./result/RSGD/fault/same_attack/q8/" + s1 + "/result_profile.out", sort="tottime")
# p = pstats.Stats("./result/RSGD/fault/same_attack/q8/" + s1 + "/result_profile.out")
# p.strip_dirs().sort_stats('tottime').print_stats(0.2)
