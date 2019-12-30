import numpy as np
import random
import matplotlib.pyplot as plt
import time
#from sklearn.model_selection import StratifiedShuffleSplit
# np.set_printoptions(threshold='nan')

class Arguments():
    def __init__(self):
        self.seed = 3 
        self.batch_size = 128 
        self.lr = 0.1
        self.lbda = 0.00 #lambda 0.001
        self.num_iter = 500 
        self.epsilon = 1e-7 
        self.num_samples = 581012 
        self.num_machines = 10 # M
        self.num_feature = 54 # n 
        self.scale = 20 # gap of computing loss
        self.D = 10

args = Arguments()
np.random.seed(args.seed)

def sigmoid(x):

    return np.exp(np.fmin(x, 0)) / (1.0 + np.exp(-np.abs(x)))

def cal_grad(w, x, y, lbda):

    # x: traing sample, shape(N, feature)
    # y: shape(N, 1)
    # w: shape(feature, 1)
    # ldba: l2-regularization parameter

    w_x_mul = np.dot(x, w)
    h_w = sigmoid(w_x_mul)
    d = y - h_w
    grad = -np.dot(x.T, d) / x.shape[0] + lbda * w
    loss = -sum(y*np.log(h_w + args.epsilon) + (1-y)*np.log(1-h_w + args.epsilon)) / x.shape[0] + lbda * sum(w ** 2) / 2.0
    return grad, loss

def cal_mean(grad_li):
    m = len(grad_li)
    grad = np.zeros_like(grad_li[0])
    grad = sum(grad_li)
    grad = grad / m
    return grad

class Machine:
    # define worker class
    def __init__(self, data_x, data_y, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(args.num_samples/args.num_machines, args.num_feature)
        data_x : a numpy array has shape :(args.num_samples/args.num_machines, args.num_feature)
        data_y: the label of the data_x, a numpy array has shape of (args.num_samples/args.num_machines, 1)"""

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id

    def update(self, theta, batch_size):
        """Calculates gradient with a randomly selected sample batch, given the current theta
         Accepts theta, a np array with shape of (args.num_feature,1)
         Returns the calculated gradient"""
        m = self.data_x.shape[0]
        # print("m:", m)
        # print batch_size
        np.random.seed(args.seed)
        if batch_size >= m:
            batch_size = m
        id = random.randint(0, m - batch_size)
        grad_f, _ = cal_grad(theta, self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], args.lbda)
        return grad_f

class Parameter_server:
    # define server class
    def __init__(self):
        """Initializes all machines"""
        self.old_theta = [] #list that stores D theta
        self.acc_li = []
        self.grad_li = []
        self.grad_norm = []
        self.loss_li = []
        self.comm_cost = []
        self.comm_cost.append(0)
        
        self.temp_comm = []
        self.temp_comm.append(0)  # initial communication point
        
        self.work_comm = [0]*args.num_machines   # communication of each worker
        
        self.pre_grad = [np.zeros((args.num_feature, 1)) for _ in range(args.num_machines)]

        path_data = "./data/covtype/"
        self.x = np.load(path_data + "x.npy")
        self.y = np.load(path_data + "y.npy").reshape(-1,1)


        samples_per_machine = args.num_samples // args.num_machines
        self.machines = []
        #########  divided dataset into workers
        for i in range(args.num_machines):
            new_machine = Machine(self.x[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  self.y[i * samples_per_machine:(i + 1) * samples_per_machine, :], i)
            self.machines.append(new_machine)


    def broadcast(self, theta, old_theta, pre_grad, alpha, batch_size, control_size, D, step, censor_type):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(args.num_featres, 1)
        Return a list containing the updated gradient of each machine with length of 'num_machines'
        theta: optimization  variable
        old_theta: stores D old theta
        pre_grad: a list of each worker's last used gradient
        alpha: step size
        step: iterations
        D: 
        type:0 - CSGD; 1 - SGD; 2 - LAG
        """

        new_grad_li = []
        comm = self.temp_comm[-1]
        
        # compute tao
        
        if censor_type ==1:
            tao = 0.0   # SGD
        else:
            if len(old_theta) < D:
                if censor_type == 0:
                    tao = control_size/(args.num_machines**2)
                else: 
                    tao=0
            else:
                temp_diff = []
                for j in range(1, len(old_theta)):
                    temp_diff.append(np.sum((old_theta[j] - old_theta[j - 1]) ** 2))
                if censor_type == 0:
                    tao = (sum(temp_diff) / (alpha**2*60 *D) + control_size)/(args.num_machines**2)
                else:
                    tao = sum(temp_diff) / (alpha**2*60 *D*args.num_machines**2)
                #print "step:", step, "temp_diff:", temp_diff
        print("step:", step, "tao:", tao)

        # compute batch size
        # batch_size = (step + 1) / (alpha * 2)
        batch_size = int(batch_size)
        # batch_size = int(batch_size)
        # batch_size = 128   # fixed batch size
        for i, mac in enumerate(self.machines):
            cur_grad = mac.update(theta, batch_size)
            grad_diff = np.sum((cur_grad - pre_grad[i]) ** 2)
            # grad_diff = np.sqrt(grad_diff)
            # print "mac", i, "diff:", grad_diff
            if grad_diff > tao:
                # print "step:", step, "mac:", i
                new_grad_li.append(cur_grad)
                pre_grad[i] = cur_grad
                comm += 1    # communicate
                print("step:", step, "mac:", i)
            else:
                new_grad_li.append(pre_grad[i])
        if step % args.scale == 0:
            self.comm_cost.append(comm)
        self.temp_comm.append(comm)
        return new_grad_li
            

    def train(self, init_theta, alpha, D):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""
        
        # print "len pre_grad:", len(self.pre_grad)
        self.old_theta.append(init_theta)
        theta = init_theta
        alpha = 0.1
        d = 0.01
        eta1=1.05
        sigma0=150
        eta2=0.99
        batch_size=eta1
        control_size=sigma0*eta2
        for i in range(args.num_iter):
            rec_grad_li = self.broadcast(theta, self.old_theta, self.pre_grad, alpha, batch_size, control_size,  D, i, 0)
            mean_grad = cal_mean(rec_grad_li)
            theta = theta - alpha * mean_grad
            # print len(self.old_theta)
            if len(self.old_theta) < D:
                self.old_theta.append(theta)
            else:
                self.old_theta.pop(0)
                self.old_theta.append(theta)
            if (i + 1) % args.scale == 0:
                _, loss = cal_grad(theta, self.x, self.y, args.lbda)
                self.loss_li.append(loss)
                print("loss: ", loss)
            batch_size=batch_size*eta1
            control_size=control_size*eta2
        print("comm cost:", self.temp_comm[-1])
        print("train end!")

    def plot_curve(self):
        """plot the loss curve
        save the learned theta to a numpy array"""

        s1 = 'alpha0.1_D10_step'
        path = "./result/covtype/"
        np.save(path + s1 + '/loss.npy', self.loss_li)

        plt.plot(np.arange(len(self.loss_li)) * args.scale, self.loss_li)
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.savefig(path + s1 + '/loss.png')
        plt.show()
        
        self.comm_cost.pop(0)
        np.save(path + s1 + '/comm.npy', self.comm_cost)

        plt.plot(self.comm_cost, self.loss_li)
        plt.xlabel('communication cost')
        plt.ylabel('loss')
        plt.savefig(path + s1 + '/comm.png')
        plt.show()


def init():
    server = Parameter_server()
    return server

def main():
    server = init()
    init_theta = np.zeros((args.num_feature, 1))
    server.train(init_theta, args.lr, args.D)
    server.plot_curve()

if __name__ == "__main__":
    main()