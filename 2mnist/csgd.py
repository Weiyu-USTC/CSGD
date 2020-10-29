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

censor_type = 0 ### type:0 - CSGD; 1 - SGD; 2 - LAG; 3 - local sgd
num_iter = 500
eta1 = 1.01
alpha = 0.1
D = 10
sigma0 = .7
eta2 = 0.991
max_batch = (num_train / num_machines) / 6

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
        if batch_size >= m:
            # batch_size = m
            idx = [random.randint(0, m-1) for i in range(0,batch_size)]
            grad_f = cal_total_grad(self.data_x[idx], self.data_y[idx], theta)
            # grad_f = cal_total_grad(self.data_x, self.data_y, theta)
        else:
            # begin = random.randint(0, m)
            idx = [random.randint(0, m-1) for i in range(0,batch_size)]
            # idx = [i % m for i in range(begin, begin + batch_size)]
            grad_f = cal_total_grad(self.data_x[idx], self.data_y[idx], theta)
        # idx = [random.randint(0, m-1) for i in range(0, batch_size)]
        # grad_f = cal_total_grad(self.data_x[idx], self.data_y[idx], theta)
        return grad_f


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.old_theta = [] #list that stores each theta, grows by one iteration
        self.acc_li = []
        self.grad_li = []
        self.grad_norm = []
        self.loss_li = []
        self.comm_cost = []
        self.comm_cost.append(0)
        
        self.temp_comm = []
        self.temp_comm.append(0)
        self.comm_event = [[0] for _ in range(num_machines)]
        
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
        samples_per_machine = int(samples_per_machine)

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
        if censor_type == 1:
            tao = -1e-8   # SGD
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
                    tao = (sum(temp_diff) / (alpha**2 * 60 *D) + control_size)/(num_machines**2)
                else:
                    tao = sum(temp_diff) / (alpha**2 * 60 * D * num_machines**2)

        for i, mac in enumerate(self.machines):
            cur_grad = mac.update(theta, int(batch_size))
            grad_diff = np.sum((cur_grad - pre_grad[i]) ** 2)
            if grad_diff > tao:
                new_grad_li.append(cur_grad)
                pre_grad[i] = cur_grad
                comm += 1
                self.comm_event[i].append(1)
                # print("step:", step, "mac:", i)
            else:
                new_grad_li.append(pre_grad[i])
                self.comm_event[i].append(0)
                # print("step:", step, "tao:", tao)
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
        batch_size = 50
        control_size = sigma0 * (num_train / num_machines) ** 2 
        # there's a scaling, which should be equivalently put in the gradients.
        for i in range(num_iter):
            rec_grad_li = self.broadcast(theta, self.old_theta, self.pre_grad, alpha, batch_size, control_size,  D, i, censor_type)
            mean_grad = cal_mean(rec_grad_li)
            theta = theta - alpha * mean_grad
            if len(self.old_theta) < D:
                self.old_theta.append(theta)
            else:
                self.old_theta.pop(0)
                self.old_theta.append(theta)
            if (i + 1) % 2 == 0:
                err = 1 - cal_acc(self.test_img_bias, self.test_lbl, theta)
                self.acc_li.append(err)
                # log(err)
                if err < 0.1:
                    break
            batch_size = min(max_batch, batch_size * eta1)
            control_size = control_size * eta2
            log("step:", i, " comm:", self.temp_comm[-1])
            
        print("comm cost:", self.temp_comm[-1])
        print("train end!")
            

    def train_localSGD(self, init_theta, alpha, LOCAL_SGD_TIME):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""
        
        theta = init_theta
        log(theta.shape)
        self.acc_li.append(1 - cal_acc(self.test_img_bias, self.test_lbl, theta))
        comm = 0
        self.comm_cost.append(comm)

        batch_size = 50
        local_theta = []
        for i, mac in enumerate(self.machines):
            local_theta.append(theta.copy())

        for i in range(num_iter):
            # theta_li = self.broadcast_localSGD(theta, alpha, batch_size, i, LOCAL_SGD_TIME)
            # theta = cal_mean(theta_li)
            if i % LOCAL_SGD_TIME != 0:
                for j, mac in enumerate(self.machines):
                    cur_grad = mac.update(local_theta[j], int(batch_size))
                    local_theta[j] = local_theta[j] - alpha * cur_grad
                    self.comm_event[j].append(0)
            else:
                for j, mac in enumerate(self.machines):
                    cur_grad = mac.update(theta, int(batch_size))
                    local_theta[j] = theta - alpha * cur_grad
                    comm += 1
                    self.comm_event[j].append(1)
            theta = cal_mean(local_theta)
            if (i + 1) % 2 == 0:
                self.comm_cost.append(comm)
                err = 1 - cal_acc(self.test_img_bias, self.test_lbl, theta)
                self.acc_li.append(err)
                if err < 0.1:
                    break
            if i % 50 == 0:
                log("step:", i)
            batch_size = min(max_batch, batch_size * eta1)
            log('step:',i,'batch_size',batch_size)
        log("comm cost:", comm)
        log("train end!")

    def plot_curve(self, s1 = 'alpha2e-5_D10'):
        if not os.path.exists('./result/' + s1):
            os.makedirs('./result/' + s1)

        np.save('./result/' + s1 + '/' + str(censor_type) + 'acc.npy', self.acc_li)        
        self.comm_cost.pop(0)
        np.save('./result/' + s1 + '/' + str(censor_type) + 'comm.npy', self.comm_cost)
        #np.save('./result/' + s1 + '/comm-event.npy', self.comm_event)

        plt.plot(self.comm_cost, self.acc_li)
        plt.xlabel('communication cost')
        plt.ylabel('error')
        # plt.title(s1)
        plt.savefig('./result/' + s1 + '/' + str(censor_type) + 'comm.png')
        plt.show()

def init():
    server = Parameter_server()
    return server

def main():
    init_theta = np.load('./init_theta.npy') # np.zeros((num_class, num_feature + 1))
    np.random.seed(3)
    server = init()
    ### CSGD/SGD/LAG/local: 0/1/2/3
    if censor_type < 3:
        server.train(init_theta, alpha, D)
    else: # local sgd 
        server.train_localSGD(init_theta, alpha, D)
    server.plot_curve()

    # for alpha in alphas:
    #     for LOCAL_SGD_TIME in LOCAL_SGD_TIMEs:
    #         np.random.seed(3)
    #         server = init()
    #         init_theta = np.zeros((num_class, num_feature + 1))
            # alpha = 0.01
            # D = 10
            # server.train(init_theta, alpha, D)
            # server.plot_curve()

            # server.train_localSGD(init_theta, alpha, LOCAL_SGD_TIME)
            # server.plot_curve(f'LocalSGD-alpha-{alpha}-TIME-{LOCAL_SGD_TIME}')


main()