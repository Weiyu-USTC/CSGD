import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os

# ====================================================
def log(*k, **kw):
    timeStamp = time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)

### parameters
num_machines = 10
num_iter = 100
num_iter_local = 100
dimension = 10
censor_type = 0 ### type:0 - CSGD; 1 - SGD; 2 - LAG; 3 - local sgd
alpha = 0.2
alpha_local = 0.2
eta1 = 1.2
sigma0 = 5
eta2 = 0.85
D = 10

max_batch = 100000

###
def CalTotalGrad(A, b, x):
    grad = np.zeros_like(x)
    m = A.shape[0]
    temp = np.dot(A, x) - b
    grad = np.sum(np.tile(temp, 10) * A, axis=0).reshape(A.shape[1], 1) / m
    return grad

def create(row, col, hetero):
    A = np.random.normal(0, 1.0, size=(row, col))
    if hetero != 0:
    	A += np.ones((row, col)) * 1
    y = np.load('./data/y.npy')
    noise = np.random.normal(0, 0.001, size=(row, 1))
    b = np.dot(A, y) + noise
    
    return A, b


def cal_mean(grad_li):
    return sum(grad_li) / len(grad_li)


class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id

    def update(self, x, batch_size, i):
        hetero = (i < 1)
        A, b = create(batch_size, dimension, hetero)
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
        ### type:0 - CSGD; 1 - SGD; 2 - LAG

        new_grad_li = []
        comm = self.comm_cost[-1]

        if censor_type == 1:
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
        # print("step:", step, "tao:", tao)

        batch_size = int(batch_size)
        # print "step:", "batch:", batch_size
        for i, mac in enumerate(self.machines):
            cur_grad = mac.update(x, batch_size, i)
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
                print("censor step:", step, "mac:", i)
        self.comm_cost.append(comm)
        return new_grad_li

    def train(self, init_x, alpha, D):
    
        self.old_x.append(init_x)
        x_star = np.load('./data/y.npy')
        self.x_li.append(init_x)
        self.x_star_norm.append(np.linalg.norm(init_x - x_star))
        self.comm_cost.append(0)
        x = init_x
        batch_size = eta1
        control_size = sigma0*eta2
        for i in range(num_iter):
            grad_li = self.broadcast(x, self.old_x, self.pre_grad, alpha, batch_size, control_size,  D, i, censor_type)
            mean_grad = cal_mean(grad_li)
            # print "mean shape:", mean_grad.shape
            x = x - alpha * mean_grad
            self.x_li.append(x)
            x_norm = np.linalg.norm(x - x_star)
            self.x_star_norm.append(x_norm)
            if x_norm < 1e-6:
                break
            batch_size = min(max_batch, batch_size * eta1)
            log(i, batch_size)
            control_size = control_size * eta2
            # print "step:", i, "grad_norm:", np.linalg.norm(mean_grad)
            # print "step:", i, "x_star_norm:", self.x_star_norm[-1]
        print("comm cost:", self.comm_cost[-1])
        print("train end!")

    def plot_curve(self, s1 = 'D10_alpha0.2'):
        if not os.path.exists('./result/' + s1):
            os.makedirs('./result/' + s1)

        np.save('./result/' + s1 + '/' + str(censor_type) + 'x_li.npy', self.x_li)
        np.save('./result/' + s1 + '/' + str(censor_type) + 'comm-event.npy', self.comm_event)

        fig = plt.figure(1)

        np.save('./result/' + s1 + '/' + str(censor_type) + 'x_star_norm.npy', self.x_star_norm)
        plt.semilogy(np.arange(len(self.x_star_norm)), self.x_star_norm)
        plt.xlabel('iter')
        plt.ylabel('||x - x*||')
        plt.title(s1)
        plt.savefig('./result/' + s1 + '/' + str(censor_type) + 'x_star_norm.png')
        plt.show()

        self.comm_cost.pop(0)
        np.save('./result/' + s1 + '/' + str(censor_type) + 'comm.npy', self.comm_cost)

        plt.semilogy(self.comm_cost, self.x_star_norm)
        plt.xlabel('communication cost')
        plt.ylabel('||x - x*||')
        plt.title(s1)
        plt.savefig('./result/' + s1 + '/' + str(censor_type) + 'comm.png')
        plt.show()

    def train_LocalSGD(self, init_x, alpha, LOCAL_SGD_TIME):
    
        self.old_x.append(init_x)
        x_star = np.load('./data/y.npy')
        self.x_li.append(init_x)
        self.x_star_norm.append(np.linalg.norm(init_x - x_star))
        self.comm_cost.append(0)
        x = init_x
        batch_size = eta1
        local_x = []
        for i, mac in enumerate(self.machines):
        	local_x.append(x.copy())

        comm = 0
        for i in range(num_iter_local):
            if i % LOCAL_SGD_TIME != 0:
            	for j, mac in enumerate(self.machines):
            		cur_grad = mac.update(local_x[j], int(batch_size), j)
            		local_x[j] = local_x[j] - alpha * cur_grad
            		self.comm_event[j].append(0)
            else:
            	for j, mac in enumerate(self.machines):
            		cur_grad = mac.update(x, int(batch_size), j)
            		local_x[j] = x - alpha * cur_grad
            		comm += 1
            		self.comm_event[j].append(1)
            x = cal_mean(local_x)
            self.comm_cost.append(comm)
            self.x_li.append(x)
            x_norm = np.linalg.norm(x - x_star)
            self.x_star_norm.append(x_norm)
            if i % 50 == 0:
            	log("step:", i)
            batch_size = min(max_batch, batch_size * eta1)
            if x_norm < 1e-6:
                break
        log("comm cost:", self.comm_cost[-1])
        log("train end!")

def init():
    server = Parameter_server()
    return server

def main():
    # LOCAL_SGD_TIMEs = [10, 30, 40]
    init_x = np.load('./data/y.npy') + np.ones((dimension,1))*0.01#np.zeros((dimension, 1))
    ### CSGD/SGD/LAG/local: 0/1/2/3
    if censor_type < 3:
    	server = init()
    	server.train(init_x, alpha, D)
    	server.plot_curve()
    else: # local sgd 
        server = init()
        server.train_LocalSGD(init_x, alpha_local, D)
        server.plot_curve()
        # for LOCAL_SGD_TIME in LOCAL_SGD_TIMEs:
            # log(f'[Begin] LocalSGD-alpha-{alpha}-TIME-{LOCAL_SGD_TIME}')
            # server.plot_curve(f'LocalSGD-alpha-{alpha}-TIME-{LOCAL_SGD_TIME}')


main()
