import numpy as np
import matplotlib.pyplot as plt
import random

#num_samples = 1000
num_machines = 10
num_iter = 170
dimension = 10

max_batch=100000

def CalTotalGrad(A, b, x):
    grad = np.zeros_like(x)
    m = A.shape[0]
    temp = np.dot(A, x) - b
    grad = np.sum(np.tile(temp, 10) * A, axis=0).reshape(A.shape[1], 1) / m
    return grad

def create(row, col):
    A = np.random.normal(0, 1.0, size=(row, col))
    y = np.load('./data/y.npy')
    noise = np.random.normal(0, 0.001, size=(row, 1))
    b = np.dot(A, y) + noise
    
    return A, b


def cal_mean(grad_li):
    m = len(grad_li)
    grad = np.zeros_like(grad_li[0])
    grad = sum(grad_li)
    # grad = grad / m
    return grad


class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id

    def update(self, x, batch_size):
        
        A, b = create(batch_size, dimension)
        grad = CalTotalGrad(A, b, x)
        return grad
    

class Parameter_server:
    def __init__(self):

        self.x0_li = []
        self.x_li = []
        self.grad_li = []
        self.grad_norm = []
        self.x_star_norm = []
        
        self.old_x = []
        self.temp_comm = []
        self.temp_comm.append(0)
        self.comm_cost = []
        self.comm_cost.append(0)

        self.comm_event = [[0] for _ in range(num_machines)]

        self.pre_grad = [np.zeros((dimension, 1)) for _ in range(num_machines)]
        
        self.machines = []
        for i in range(num_machines):
            new_machine = Machine(i)
            self.machines.append(new_machine)

    def broadcast(self, x, old_theta, pre_grad, alpha, batch_size, control_size, D, step, censor_type):

        #type:0 - CSGD; 1 - SGD; 2 - LAG

        new_grad_li = []
        comm = self.comm_cost[-1]

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
        # print "step:", "batch:", batch_size
        for i, mac in enumerate(self.machines):
            cur_grad = mac.update(x, batch_size)
            grad_diff = np.sum((cur_grad - pre_grad[i]) ** 2)
            if grad_diff > tao:
                new_grad_li.append(cur_grad)
                pre_grad[i] = cur_grad
                comm += 1
                self.comm_event[i].append(1)
                print "step:", step, "mac:", i
            else:
                new_grad_li.append(pre_grad[i])
                self.comm_event[i].append(0)
        # if step % 2 == 0:
        self.comm_cost.append(comm)
        # self.temp_comm.append(comm)
        return new_grad_li

    def train(self, init_x, alpha, D):
    
        self.old_x.append(init_x)
        x_star = np.load('./data/y.npy')
        self.x_li.append(init_x)
        x = init_x
        eta1=1.1
        sigma0=.1
        eta2=.91
        batch_size=eta1
        control_size=sigma0*eta2
        for i in range(num_iter):
            grad_li = self.broadcast(x, self.old_x, self.pre_grad, alpha, batch_size, control_size,  D, i, 0)
            mean_grad = cal_mean(grad_li)
            # print "mean shape:", mean_grad.shape
            x = x - alpha * mean_grad
            self.x_li.append(x)
            self.x_star_norm.append(np.linalg.norm(x - x_star))
            if batch_size>max_batch:
                batch_size=max_batch
            else:
                batch_size=batch_size*eta1
            control_size=control_size*eta2
            #print "step:", i, "grad_norm:", np.linalg.norm(mean_grad)
            #print "step:", i, "x_star_norm:", self.x_star_norm[-1]
        print("comm cost:", self.comm_cost[-1])
        print("train end!")

    def plot_curve(self):

        s1 = 'D10_alpha0.02'

        np.save('./result/' + s1 + '/x_li.npy', self.x_li)
        np.save('./result/' + s1 + '/comm-event.npy', self.comm_event)

        fig = plt.figure(1)
        # plt.semilogy(np.arange(num_iter), self.grad_norm)
        # plt.xlabel('iter')
        # plt.ylabel('log(||grad||)')
        # plt.title(s1)
        # plt.savefig('./result/RDSGD/no_fault/20/' + s1 + '/grad_norm.jpg')
        # plt.show()

        np.save('./result/' + s1 + '/x_star_norm.npy', self.x_star_norm)
        plt.semilogy(np.arange(num_iter), self.x_star_norm)
        plt.xlabel('iter')
        plt.ylabel('log||x - x*||')
        plt.title(s1)
        plt.savefig('./result/' + s1 + '/x_star_norm.jpg')
        plt.show()

        self.comm_cost.pop(0)
        #print len(self.comm_cost)
        #print len(self.x_star_norm)
        np.save('./result/' + s1 + '/comm.npy', self.comm_cost)

        plt.semilogy(self.comm_cost, self.x_star_norm)
        plt.xlabel('communication cost')
        plt.ylabel('log||x - x*||')
        # plt.title(s1)
        plt.savefig('./result/' + s1 + '/comm.png')
        plt.show()


def init():
    server = Parameter_server()
    return server

def main():
    server = init()
    init_x = np.zeros((dimension, 1))
    # init_x = []
    # for i in range(num_machines):
    #     init_x.append(np.zeros((dimension, )))
    alpha = 0.02
    D = 10
    server.train(init_x, alpha, D)
    server.plot_curve()

main()

# A = np.arange(12).reshape(4,3)
# x = np.arange(3).reshape(3,1)
# b = np.arange(4).reshape(4,1)
# temp = (np.dot(A, x) - b)
# print temp
# print A.shape
# print np.tile(temp, A.shape[1])*A