# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 基本定义

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import pickle
import traceback


# %%
import torch
import torchvision
import matplotlib.pyplot as plt

# %% [markdown]
# ## 数据集/模型

# %%
# ResNet50 + CIFAR100
# optConfig = {
#     'rounds': 10,
#     'displayInterval': 5000,
#     'decreaseInterval': 500,
    
#     'weight_decay': 0.0001,
    
#     'fixSeed': True,
#     'SEED': 100,
    
#     'initBatchSize': 16,
#     'maxBatchSize': 128,
#     'eta1': 1.05,

#     'shuffle': False,
    
#     'nodeSize': 10,
#     'D': 10,
    
#     'loader_num_worker': 0,
#     'loader_pin_memory': False,
    
#     'gamma': 0.1,
# }

# # 加载数据集
# preprocess = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomCrop(32, padding=4),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         mean=[n/255. for n in [129.3, 124.1, 112.4]], 
#         std=[n/255. for n in [68.2,  65.4,  70.4]]
#     )
# ])

# train_dataset = torchvision.datasets.CIFAR100(root='./dataset/',
#                                              train=True, 
#                                              transform=preprocess,
#                                              download=False)

# validate_dataset = torchvision.datasets.CIFAR100(root='./dataset/',
#                                             train=False, 
#                                             transform=preprocess)

# # 模型工厂
# def modelFactory(fixSeed=False, SEED=100):
#     model = torchvision.models.resnet18(
#         num_classes=len(train_dataset.classes)
#     )
#     if fixSeed:
#         FILE_NAME = f'resnet18_init_class_{len(train_dataset.classes)}.pkl'
#         if os.path.exists(FILE_NAME):
#             model.load_state_dict(torch.load(FILE_NAME))
#         else:
#             torch.save(model.state_dict(), FILE_NAME)
#     return model

# %% [markdown]
# ### ResNet + CIFAR10

# %%
# ResNet18 + CIFAR10
optConfig = {
    'rounds': 10,
    'displayInterval': 6000,
    'decreaseInterval': 500,
    
    'weight_decay': 0.0001,
    
    'fixSeed': False,
    'SEED': 100,
    
    'initBatchSize': 16,
    'maxBatchSize': 64,
    'eta1': 1.05,
    'shuffle': True,
    
    'nodeSize': 10,
    'D': 10,
    
    'loader_num_worker': 0,
    'loader_pin_memory': False,
    
    'gamma': 3e-1,
}

# 加载数据集
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = torchvision.datasets.CIFAR10(root='./dataset/',
                                             train=True, 
                                             transform=preprocess,
                                             download=False)
validate_dataset = torchvision.datasets.CIFAR10(root='./dataset/',
                                            train=False, 
                                            transform=preprocess)

# 模型工厂
def modelFactory(fixSeed=False, SEED=100):
    model = torchvision.models.resnet18(
        num_classes=len(train_dataset.classes)
    )
    if fixSeed:
        FILE_NAME = f'resnet18_init_class_{len(train_dataset.classes)}.pkl'
        if os.path.exists(FILE_NAME):
            model.load_state_dict(torch.load(FILE_NAME))
        else:
            torch.save(model.state_dict(), FILE_NAME)
    return model


# %%
# # ResNet50 + CIFAR10_superclass
# optConfig = {
#     'rounds': 10,
#     'displayInterval': 5000,
#     'decreaseInterval': 500,
    
#     'weight_decay': 0.0001,
    
#     'fixSeed': False,
#     'SEED': 100,
    
#     'initBatchSize': 16,
#     'maxBatchSize': 64,
#     'shuffle': True,
    
#     'nodeSize': 10,
#     'D': 10,
    
#     'loader_num_worker': 0,
#     'loader_pin_memory': False,
    
#     'gamma': 3e-1,
# }

# class CIFAR100_superclass(torchvision.datasets.CIFAR100):
#     def __init__(self, *args, **kw):
#         super(CIFAR100_superclass, self).__init__(*args, **kw)
        
#         new_class_to_idx = {
#             "beaver": 0, "dolphin": 0, "otter": 0, "seal": 0, "whale": 0, 
#             "aquarium_fish": 1, "flatfish": 1, "ray": 1, "shark": 1, "trout": 1, 
#             "orchid": 2, "poppy": 2, "rose": 2, "sunflower": 2, "tulip": 2, 
#             "bottle": 3, "bowl": 3, "can": 3, "cup": 3, "plate": 3, 
#             "apple": 4, "mushroom": 4, "orange": 4, "pear": 4, "sweet_pepper": 4, 
#             "clock": 5, "keyboard": 5, "lamp": 5, "telephone": 5, "television": 5, 
#             "bed": 6, "chair": 6, "couch": 6, "table": 6, "wardrobe": 6, 
#             "bee": 7, "beetle": 7, "butterfly": 7, "caterpillar": 7, "cockroach": 7, 
#             "bear": 8, "leopard": 8, "lion": 8, "tiger": 8, "wolf": 8, 
#             "bridge": 9, "castle": 9, "house": 9, "road": 9, "skyscraper": 9, 
#             "cloud": 10, "forest": 10, "mountain": 10, "plain": 10, "sea": 10, 
#             "camel": 11, "cattle": 11, "chimpanzee": 11, "elephant": 11, "kangaroo": 11, 
#             "fox": 12, "porcupine": 12, "possum": 12, "raccoon": 12, "skunk": 12, 
#             "crab": 13, "lobster": 13, "snail": 13, "spider": 13, "worm": 13, 
#             "baby": 14, "boy": 14, "girl": 14, "man": 14, "woman": 14, 
#             "crocodile": 15, "dinosaur": 15, "lizard": 15, "snake": 15, "turtle": 15, 
#             "hamster": 16, "mouse": 16, "rabbit": 16, "shrew": 16, "squirrel": 16, 
#             "maple_tree": 17, "oak_tree": 17, "palm_tree": 17, "pine_tree": 17, "willow_tree": 17, 
#             "bicycle": 18, "bus": 18, "motorcycle": 18, "pickup_truck": 18, "train": 18, 
#             "lawn_mower": 19, "rocket": 19, "streetcar": 19, "tank": 19, "tractor": 19, 
#         }
#         self.class_map = [new_class_to_idx[c] for c in self.classes]
#         self.classes = [
#             "aquatic mammals", "fish", "flowers	orchids", "food containers",
#             "fruit and vegetables", "household electrical devices", "household furniture", "insects", 
#             "large carnivores", "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
#             "medium-sized mammals", "non-insect invertebrates", "people", "reptiles",
#             "small mammals", "trees", "vehicles 1", "vehicles 2"
#         ]
#         self.class_to_idx = None
#     def __getitem__(self, idx):
#         materials, labels_crude = torchvision.datasets.CIFAR100.__getitem__(self, idx)
#         labels_list = [self.class_map[label] for label in labels_crude]
#         labels = torch.tensor(labels_list)
#         return materials, labels

# train_dataset = CIFAR100_superclass(root='./dataset/',
#                                              train=True, 
#                                              transform=preprocess,
#                                              download=False)
# validate_dataset = CIFAR100_superclass(root='./dataset/',
#                                             train=False, 
#                                             transform=preprocess)


# # 模型工厂
# def modelFactory(fixSeed=False, SEED=100):
#     model = torchvision.models.resnet18(
#         num_classes=len(train_dataset.classes)
#     )
#     if fixSeed:
#         FILE_NAME = f'resnet18_init_class_{len(train_dataset.classes)}.pkl'
#         if os.path.exists(FILE_NAME):
#             model.load_state_dict(torch.load(FILE_NAME))
#         else:
#             torch.save(model.state_dict(), FILE_NAME)
#     return model


# %%
# import torchsummary
# torchsummary.summary(model, input_size=(3, 32, 32))

# %% [markdown]
# ## 数据集属性统计

# %%
assert type(train_dataset) == type(validate_dataset)

MAX_FEATURE = 1
for d in train_dataset.data.shape[1:]:
    MAX_FEATURE *= d

# 数据集属性
dataSetConfig = {
    'name': type(train_dataset).__name__,

    'dataSetSize': train_dataset.data.shape[0],
    'maxFeature': MAX_FEATURE,
    
    'evaluationBatchSize': 200,
}

# %% [markdown]
# ## 数据分布

# %%
def heterogeneousPartition(dataset, nodeSize):
    dataPartitionTable = [[] for _ in range(nodeSize)]
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        allocateTo = label % nodeSize
        dataPartitionTable[allocateTo].append(idx)
    return dataPartitionTable

def defaultPartition(nodeSize):
    pieces = [(i*len(train_dataset)) // nodeSize for i in range(nodeSize+1)]
    dataPartition = [list(range(pieces[i], pieces[i+1])) for i in range(nodeSize)]
    return dataPartition


# %%
# 普通分布
dataPartition = defaultPartition(optConfig['nodeSize'])
# --------------------------------------------------
# 异构分布
# dataPartition = heterogeneousPartition(train_dataset, optConfig['nodeSize'])
# dataSetConfig['name'] += '_hetero'
# --------------------------------------------------


# 数据分发
train_dataset_subset = [
    torch.utils.data.Subset(train_dataset, dataPartition[node])
    for node in range(optConfig['nodeSize'])
]
    

# %% [markdown]
# ## 运行参数

# %%
CACHE_DIR = './cache/'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


# %%
device = 'cuda:0'

# %% [markdown]
# ## 辅助函数

# %%
# 报告函数
def log(*k, **kw):
    timeStamp = time.strftime('[%m-%d %H:%M:%S] ', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)
def debug(*k, **kw):
    timeStamp = time.strftime('[%m-%d %H:%M:%S] (debug)', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)

# %% [markdown]
# ## 损失函数

# %%
loss_func = torch.nn.CrossEntropyLoss()
INIT_TRAIN_LOSS = math.log2(len(train_dataset.classes))
INIT_TRAIN_ACC = 1/len(train_dataset.classes)


# %%
# 顺序遍历loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=optConfig['maxBatchSize'], shuffle=False)
validate_loader = torch.utils.data.DataLoader(
    dataset=validate_dataset, batch_size=optConfig['maxBatchSize'], shuffle=False
)

def calculateAccuracy(model, loader, device):
    loss = 0
    accuracy = 0
    total = 0
    
    model = model.to(device)
    
    with torch.no_grad():
        for material, targets in loader:
            material, targets = material.to(device), targets.to(device)
            outputs = model(material)

            l = loss_func(outputs, targets)

            loss += l.item() * len(targets)
            _, predicted = torch.max(outputs.data, dim=1)
            accuracy += (predicted == targets).sum().item()
            total += len(targets)
    
    loss /= total
    accuracy /= total
    
    return loss, accuracy

# %% [markdown]
# ## torch辅助函数

# %%
def fixSeed(SEED):
    torch.manual_seed(SEED)
    try:
        np.random.seed(SEED)
    except Exception as _:
        pass
    try:
        numpy.random.seed(SEED)
    except Exception as _:
        pass
    try:
        random.seed(SEED)
    except Exception as _:
        pass


# %%
def randomSample(dataset, batchSize):
    m, t = zip(*random.sample(dataset, batchSize))
    material, targets = torch.cat(m), torch.tensor(t)
    return material, targets


# %%
def releaseCUDA(func):
    def wrapper(*args, **kw):
        torch.cuda.empty_cache()
        func(*args, **kw)
        torch.cuda.empty_cache()
    return wrapper


# %%
def getPara(model, useString=True):
    para = sum([x.nelement() for x in model.parameters()])
    if not useString:
        return para
    elif para >= 1<<20:
        return '{:.2f}M'.format(para / (1<<20))
    elif para >= 1<<10:
        return '{:.2f}K'.format(para / (1<<10))
    else:
        return str(para)

# %% [markdown]
# # 运行记录器

# %%
# 记录器，用来在训练过程中显示中间结果
class CentralRecorder():
    def __init__(self, rounds, displayInterval):
        self.rounds = rounds
        self.displayInterval = displayInterval
        
#         # 动态画图更新进度
#         %matplotlib notebook

#         # 得到当前画布和坐标轴
#         plt.figure()
#         plt.plot([], [])
#         fig = plt.gcf()
#         ax = plt.gca()
        
#         self.fig, self.axis = plt.subplots(1, 2)
#         self.axis[0].plot([], [], label='train loss')
#         self.axis[0].plot([], [], label='validation loss')
#         self.axis[1].plot([], [], label='train accuracy')
#         self.axis[1].plot([], [], label='validation accuracy')
#         for ax in self.axis:
#             ax.legend()
    
#         # 调整画布形状
#         self.fig.set_size_inches(6, 3)
        
    def report(self, r, trainLossPath, trainAccPath, valLossPath, valAccPath):
        # 打印日志
        trainLoss, trainAcc = trainLossPath[-1], trainAccPath[-1]
        valLoss, valAcc = valLossPath[-1], valAccPath[-1]
        
        log(f'[{r}/{self.rounds}](interval: {self.displayInterval:.0f}) ' +
            f'train: loss={trainLoss:.4f} acc={trainAcc:.2f} val: loss={valLoss:.4f} acc={valAcc:.2f}'
        )
        
        # 更新图像
#         self.axis[0].lines[0].set_data(range(r+1), trainLossPath)
#         self.axis[0].lines[1].set_data(range(r+1), trainAccPath)
#         self.axis[1].lines[0].set_data(range(r+1), valLossPath)
#         self.axis[1].lines[1].set_data(range(r+1), valAccPath)
        
#         for ax in self.axis:
#             ax.relim()
#             ax.autoscale_view()
#         self.fig.canvas.draw()


# %%
# 记录器，用来在训练过程中显示中间结果，需要显示通讯信息的减少
class CommunicationRecorder():
    def __init__(self, rounds, displayInterval, nodeSize):
        self.rounds = rounds
        self.displayInterval = displayInterval
        self.nodeSize = nodeSize
        
        # 动态画图更新进度
#         %matplotlib notebook

        # 得到当前画布和坐标轴
#         plt.figure()
#         plt.plot([], [])
#         fig = plt.gcf()
#         ax = plt.gca()
        
#         self.fig, self.axis = plt.subplots(2, 2)
#         # 第一行显示metric-round图
#         self.axis[0][0].plot([], [], label='train loss')
#         self.axis[0][0].plot([], [], label='validation loss')
#         self.axis[0][1].plot([], [], label='train accuracy')
#         self.axis[0][1].plot([], [], label='validation accuracy')
        
#         # 第二行显示metric-communication图
#         self.axis[1][0].plot([], [], label='train loss')
#         self.axis[1][0].plot([], [], label='validation loss')
#         self.axis[1][1].plot([], [], label='train accuracy')
#         self.axis[1][1].plot([], [], label='validation accuracy')
        
#         for axline in self.axis:
#             for ax in axline:
#                 ax.legend()
    
#         # 调整画布形状
#         self.fig.set_size_inches(6, 6)
        
    def report(self, r, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath):
        # 打印日志
        trainLoss, trainAcc = trainLossPath[-1], trainAccPath[-1]
        valLoss, valAcc = valLossPath[-1], valAccPath[-1]
        if r == 0:
            communication = 0
            communicationRate = 1
        else:
            communication = communicationPath[-1]
            communicationRate = communication / (r*self.displayInterval*self.nodeSize)
        
        log('[{}/{}](interval: {:.0f}) train: loss={:.4f} acc={:.2f} val: loss={:.4f} acc={:.2f} com:{}({:.1f}%)'
            .format(r, self.rounds, self.displayInterval,
                    trainLoss, trainAcc, valLoss, valAcc,
                    communication, communicationRate*100)
        )
        
        # 更新图像
#         self.axis[0][0].lines[0].set_data(range(r+1), trainLossPath)
#         self.axis[0][0].lines[1].set_data(range(r+1), valLossPath)
#         self.axis[0][1].lines[0].set_data(range(r+1), trainAccPath)
#         self.axis[0][1].lines[1].set_data(range(r+1), valAccPath)
        
#         self.axis[1][0].lines[0].set_data(communicationPath, trainLossPath)
#         self.axis[1][0].lines[1].set_data(communicationPath, valLossPath)
#         self.axis[1][1].lines[0].set_data(communicationPath, trainAccPath)
#         self.axis[1][1].lines[1].set_data(communicationPath, valAccPath)
        
#         for axline in self.axis:
#             for ax in axline:
#                 ax.relim()
#                 ax.autoscale_view()
#         self.fig.canvas.draw()

# %% [markdown]
# # 优化算法
# %% [markdown]
# ## Central SGD

# %%
def CentralSGD(model, gamma, weight_decay, attack=None, 
          rounds=10, displayInterval=1000, 
          device='cpu', SEED=100, fixSeed=False, 
          batchSize=1,
          **kw):
    if fixSeed:
        random.seed(SEED)
    recorder = CentralRecorder(rounds, displayInterval)

    # 随机取样器
    randomSampler = torch.utils.data.sampler.RandomSampler(
        train_dataset, 
        num_samples=rounds*displayInterval*batchSize, 
        replacement=True
    )
    train_random_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batchSize, 
        sampler=randomSampler,
    )
    randomIter = iter(train_random_loader)
    
    # 求初始误差
    trainLoss, trainAcc = calculateAccuracy(model, train_loader, device)
    valLoss, valAcc = calculateAccuracy(model, validate_loader, device)

    trainLossPath = [trainLoss]
    trainAccPath = [trainAcc]
    valLossPath = [valLoss]
    valAccPath = [valAcc]
    
    recorder.report(0, trainLossPath, trainAccPath, valLossPath, valAccPath)

    for r in range(rounds):
        model.train()
        for k in range(displayInterval):
            # 读取数据
            material, targets = next(randomIter)
            material, targets = material.to(device), targets.to(device)

            # 随机梯度
            # --------------------
            # 预测
            outputs = model(material)
            loss = loss_func(outputs, targets)
            # 反向传播
            model.zero_grad()
            loss.backward()

            # 更新
            for para in model.parameters():
                para.data.add_(-gamma, para.grad)
                para.data.add_(-weight_decay, para)
        
        model.eval()
        trainLoss, trainAcc = calculateAccuracy(model, train_loader, device)
        valLoss, valAcc = calculateAccuracy(model, validate_loader, device)

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAcc)
        valLossPath.append(valLoss)
        valAccPath.append(valAcc)

        recorder.report(r+1, trainLossPath, trainAccPath, valLossPath, valAccPath)
    return model, trainLossPath, trainAccPath, valLossPath, valAccPath, []

# %% [markdown]
# ## SGD

# %%
SGDConfig = optConfig.copy()


# %%
def SGD(model, gamma, weight_decay, 
        eta1=1.05,               # batchSize增长率
        decreaseInterval = 2000, # 参数变化的间隔
        initBatchSize=5,         # 初始batchSize
        maxBatchSize=None,       # 最大batchSize
        rounds=10, displayInterval=1000, 
        nodeSize=0, device='cpu',
        loader_num_worker=0, loader_pin_memory=False,
        **kw):
    
    batchSize = initBatchSize
    
    # 记录器
    recorder = CommunicationRecorder(rounds, displayInterval, nodeSize)

    train_dataset_subset = [
        torch.utils.data.Subset(train_dataset, dataPartition[node])
        for node in range(nodeSize)
    ]
    
    # 求初始误差
    trainLoss, trainAcc = INIT_TRAIN_LOSS, INIT_TRAIN_ACC
    valLoss, valAcc = calculateAccuracy(model, validate_loader, device)
    # 通信次数
    communicationCost = 0
    
    trainLossPath = [trainLoss]
    trainAccPath = [trainAcc]
    valLossPath = [valLoss]
    valAccPath = [valAcc]
    communicationPath = [communicationCost]
    
    recorder.report(0, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)

    for r in range(rounds):
        # 损失记录器
        trainLossAvg = 0
        trainAccAvg = 0
        totalSample = 0
        
        for k in range(displayInterval):
            iteration = r * displayInterval + k
            
            # 创建新的loader
            if iteration % decreaseInterval == 0:
                batchSize_int = maxBatchSize if maxBatchSize != None and batchSize > maxBatchSize else int(batchSize)
                # 随机取样器
                randomSampler = lambda subset: torch.utils.data.sampler.RandomSampler(
                    subset, 
                    num_samples = decreaseInterval*batchSize_int, 
                    replacement = True
                )
                train_random_loaders_splited = [torch.utils.data.DataLoader(
                    dataset = subset,
                    batch_size = batchSize_int, 
                    sampler = randomSampler(subset),
                ) for subset in train_dataset_subset]
                randomIters = [iter(loader) for loader in train_random_loaders_splited]

            gradient = [
                torch.zeros_like(para) for para in model.parameters()
            ]
            
            # 诚实节点更新
            for node in range(nodeSize):
                # 读取数据
                material, targets = next(randomIters[node])
                material, targets = material.to(device), targets.to(device)
                
                # 随机梯度
                # --------------------
                # 预测
                outputs = model(material)
                loss = loss_func(outputs, targets)
                # 反向传播
                model.zero_grad()
                loss.backward()

                # 更新梯度
                for pi, para in enumerate(model.parameters()):
                    gradient[pi].data.add_(1 / nodeSize, para.grad.data)
                    gradient[pi].data.add_(weight_decay / nodeSize, para)
                
                # 记录通信次数
                communicationCost += 1
                # 记录准确率和损失
                _, predicted = torch.max(outputs.data, dim=1)
                trainAccAvg += (predicted == targets).sum().item()
                trainLossAvg += loss.item()
                totalSample += len(targets)
                
            # 更新
            for para, grad in zip(model.parameters(), gradient):
                para.data.add_(-gamma, grad)
                
            # 更新batchSize
            iteration = r * displayInterval + k
            if iteration != 0 and iteration % decreaseInterval == 0:
                batchSize *= eta1
        
        # 记录信息
        communicationPath.append(communicationCost)
        
        trainLoss, trainAcc = trainLossAvg / totalSample, trainAccAvg / totalSample
        valLoss, valAcc = calculateAccuracy(model, validate_loader, device)

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAcc)
        valLossPath.append(valLoss)
        valAccPath.append(valAcc)
        
        recorder.report(r+1, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)
    return model, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath

# %% [markdown]
# ## Local SGD

# %%
LocalSGDConfig = optConfig.copy()
LocalSGDConfig['D'] = 10


# %%
def LocalSGD(model, gamma, weight_decay, 
        eta1=1.05,               # batchSize增长率
        decreaseInterval = 2000, # 参数变化的间隔
        initBatchSize=5,         # 初始batchSize
        maxBatchSize=None,       # 最大batchSize
        D=10,                    # 局部执行的次数
        rounds=10, displayInterval=1000, 
        nodeSize=0, device='cpu',
        loader_num_worker=0, loader_pin_memory=False,
        **kw):
    
    # 简化算法，保证展示间隔和参数下降间隔刚好可以放下一个inner iteration
    assert displayInterval % D == 0
    assert decreaseInterval % D == 0 
    
    batchSize = initBatchSize
        
    # 记录器
    recorder = CommunicationRecorder(rounds, displayInterval, nodeSize)

    # 求初始误差
    trainLoss, trainAcc = INIT_TRAIN_LOSS, INIT_TRAIN_ACC
    valLoss, valAcc = calculateAccuracy(model, validate_loader, device)
    # 通信次数
    communicationCost = 0
    
    trainLossPath = [trainLoss]
    trainAccPath = [trainAcc]
    valLossPath = [valLoss]
    valAccPath = [valAcc]
    communicationPath = [communicationCost]
    
    recorder.report(0, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)

    # 本地模型分配内存
    localModel = modelFactory(fixSeed=False)
    localModel = localModel.to(device)
    
    for r in range(rounds):
        # 损失记录器
        trainLossAvg = 0
        trainAccAvg = 0
        totalSample = 0

        for k in range(displayInterval // D):
            # 实际iteration数乘D
            iteration = (r * displayInterval + k) * D
            
            # 创建新的loader
            if iteration % decreaseInterval == 0:
                batchSize_int = maxBatchSize if maxBatchSize != None and batchSize > maxBatchSize else int(batchSize)
                # 随机取样器
                randomSampler = lambda subset: torch.utils.data.sampler.RandomSampler(
                    subset, 
                    num_samples = decreaseInterval*batchSize_int*D, 
                    replacement = True
                )
                train_random_loaders_splited = [torch.utils.data.DataLoader(
                    dataset = subset,
                    batch_size = batchSize_int, 
                    sampler = randomSampler(subset),
                ) for subset in train_dataset_subset]
                randomIters = [iter(loader) for loader in train_random_loaders_splited]
            
            aggregatedModel = [
                torch.zeros_like(para) for para in model.parameters()
            ]
            
            # 诚实节点更新
            for node in range(nodeSize):
                # 本地节点拉取模型
                localModel.load_state_dict(model.state_dict())
                
                for innerIteration in range(D):
                    # 读取数据
                    material, targets = next(randomIters[node])
                    material, targets = material.to(device), targets.to(device)

                    # 随机梯度
                    # --------------------
                    # 预测
                    outputs = localModel(material)
                    loss = loss_func(outputs, targets)
                    # 反向传播
                    localModel.zero_grad()
                    loss.backward()

                    # 更新梯度
                    for pi, para in enumerate(localModel.parameters()):
                        para.data.sub_(weight_decay*gamma, para)
                        para.data.sub_(gamma, para.grad.data)
                
                # 提交最新模型
                for pi, para in enumerate(localModel.parameters()):
                    aggregatedModel[pi].data.add_(1 / nodeSize, para.data)
                # 记录通信次数
                communicationCost += 1
                # 记录准确率和损失
                _, predicted = torch.max(outputs.data, dim=1)
                trainAccAvg += (predicted == targets).sum().item()
                trainLossAvg += loss.item()
                totalSample += len(targets)
                
            # 更新
            for para, para_latest in zip(model.parameters(), aggregatedModel):
                para.data.copy_(para_latest)
                
            # 更新batchSize
            if iteration != 0 and iteration % decreaseInterval == 0:
                batchSize *= eta1
        
        # 记录信息
        communicationPath.append(communicationCost)
        
        trainLoss, trainAcc = trainLossAvg / totalSample, trainAccAvg / totalSample
        valLoss, valAcc = calculateAccuracy(model, validate_loader, device)

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAcc)
        valLossPath.append(valLoss)
        valAccPath.append(valAcc)
        
        recorder.report(r+1, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)
    return model, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath

# %% [markdown]
# ## CSGD

# %%
CSGDConfig = optConfig.copy()
CSGDConfig['sigma0'] = 1700
CSGDConfig['eta1'] = optConfig['eta1']
CSGDConfig['eta2'] = 0.96
CSGDConfig['D'] = 10


# %%
def CSGD(model, gamma, weight_decay, 
        sigma0=15,               # 初始control size
        eta1=1.05,               # batchSize增长率
        eta2=0.96,               # controlSize缩减率
        D=10,                    # 用于估计梯度的旧梯度个数
        decreaseInterval = 2000, # 参数变化的间隔
        initBatchSize=5,         # 初始batchSize
        maxBatchSize=None,       # 最大batchSize
        rounds=10, displayInterval=1000, 
        device='cpu', nodeSize=0, 
        loader_num_worker=0, loader_pin_memory=False,
        **kw):
    
    batchSize = initBatchSize
    controlSize = sigma0
        
    # 记录器
    recorder = CommunicationRecorder(rounds, displayInterval, nodeSize)
    
    # 求初始误差
    trainLoss, trainAcc = INIT_TRAIN_LOSS, INIT_TRAIN_ACC
    valLoss, valAcc = calculateAccuracy(model, validate_loader, device)
    # 通信次数
    communicationCost = 0
    
    trainLossPath = [trainLoss]
    trainAccPath = [trainAcc]
    valLossPath = [valLoss]
    valAccPath = [valAcc]
    communicationPath = [communicationCost]
    
    recorder.report(0, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)

    # 存储旧梯度的模长 - (nodeSize*D)
    gradientRecord = [
        [0] * D for node in range(nodeSize)
    ]
    # 发送的梯度 - (nodeSize*dimension)
    messages = [
        [torch.zeros_like(para) for para in model.parameters()]
        for node in range(nodeSize)
    ]
    
    for r in range(rounds):
        # 损失记录器
        trainLossAvg = 0
        trainAccAvg = 0
        totalSample = 0
        
        for k in range(displayInterval):
            iteration = r * displayInterval + k
            
            # 创建新的loader
            if iteration % decreaseInterval == 0:
                batchSize_int = maxBatchSize if maxBatchSize != None and batchSize > maxBatchSize else int(batchSize)
                # 随机取样器
                randomSampler = lambda subset: torch.utils.data.sampler.RandomSampler(
                    subset, 
                    num_samples = decreaseInterval*batchSize_int, 
                    replacement = True
                )
                train_random_loaders_splited = [torch.utils.data.DataLoader(
                    dataset = subset,
                    batch_size = batchSize_int, 
                    sampler = randomSampler(subset),
                ) for subset in train_dataset_subset]
                randomIters = [iter(loader) for loader in train_random_loaders_splited]

            # 诚实节点更新
            for node in range(nodeSize):
                # 读取数据
                material, targets = next(randomIters[node])
                material, targets = material.to(device), targets.to(device)
                
                # 随机梯度
                # --------------------
                # 预测
                outputs = model(material)
                loss = loss_func(outputs, targets)
                # 反向传播
                model.zero_grad()
                loss.backward()
                # 加上weight_decay
                for para in model.parameters():
                    para.grad.data.add_(weight_decay, para)

                # censor
                # --------------------
                # 理论确定的常数
                w = 1/60
                tau = (w*sum(gradientRecord[node])/D + controlSize) / (nodeSize**2)
                # 梯度变化
                diff = sum([((oldGrad-para.grad.data).norm()**2).item()
                            for (oldGrad, para) in zip(messages[node], model.parameters())])
                if diff > tau:
                    for (para, grad) in zip(model.parameters(), messages[node]):
                        grad.data.copy_(para.grad.data)
                    gradientNorm = sum([(para.grad.data.norm()**2).item() 
                                         for para in model.parameters()
                                        ])
                    gradientRecord[node].append(gradientNorm)
                    communicationCost += 1
                else:
                    gradientRecord[node].append(gradientRecord[node][-1])
                gradientRecord[node].pop(0)
                # --------------------

                # 记录准确率和损失
                _, predicted = torch.max(outputs.data, dim=1)
                trainAccAvg += (predicted == targets).sum().item()
                trainLossAvg += loss.item()
                totalSample += len(targets)
                
            # 更新
            for message in messages:
                for para, grad in zip(model.parameters(), message):
                    para.data.add_(-gamma / nodeSize, grad)
                   
            # 更新CSGD参数
            if iteration != 0 and iteration % decreaseInterval == 0:
                batchSize *= eta1
                controlSize *= eta2
        
        # 记录信息
        communicationPath.append(communicationCost)
        
        trainLoss, trainAcc = trainLossAvg / totalSample, trainAccAvg / totalSample
        valLoss, valAcc = calculateAccuracy(model, validate_loader, device)

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAcc)
        valLossPath.append(valLoss)
        valAccPath.append(valAcc)
        
        recorder.report(r+1, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)
    return model, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath

# %% [markdown]
# ## LAG-S

# %%
LAGSConfig = optConfig.copy()
LAGSConfig['sigma0'] = 2000
LAGSConfig['eta1'] = optConfig['eta1']
LAGSConfig['eta2'] = 0.96
LAGSConfig['D'] = 10


# %%
def LAG_S(model, gamma, weight_decay, 
        eta1=1.05,               # batchSize增长率
        D=10,                    # 用于估计梯度的旧梯度个数
        decreaseInterval = 2000, # 参数变化的间隔
        initBatchSize=5,         # 初始batchSize
        maxBatchSize=None,       # 最大batchSize
        rounds=10, displayInterval=1000, 
        nodeSize=0, device='cpu', 
        loader_num_worker=0, loader_pin_memory=False,
        **kw):
    
    batchSize = initBatchSize
            
    # 记录器
    recorder = CommunicationRecorder(rounds, displayInterval, nodeSize)

    # 求初始误差
    trainLoss, trainAcc = INIT_TRAIN_LOSS, INIT_TRAIN_ACC
    valLoss, valAcc = calculateAccuracy(model, validate_loader, device)
    # 通信次数
    communicationCost = 0
    
    trainLossPath = [trainLoss]
    trainAccPath = [trainAcc]
    valLossPath = [valLoss]
    valAccPath = [valAcc]
    communicationPath = [communicationCost]
    
    recorder.report(0, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)

    # 存储旧梯度的模长 - (nodeSize*D)
    gradientRecord = [
        [0] * D for node in range(nodeSize)
    ]
    # 发送的梯度 - (nodeSize*dimension)
    messages = [
        [torch.zeros_like(para) for para in model.parameters()]
        for node in range(nodeSize)
    ]
    
    for r in range(rounds):
        # 损失记录器
        trainLossAvg = 0
        trainAccAvg = 0
        totalSample = 0

        for k in range(displayInterval):
            iteration = r * displayInterval + k
            
            # 创建新的loader
            if iteration % decreaseInterval == 0:
                batchSize_int = maxBatchSize if maxBatchSize != None and batchSize > maxBatchSize else int(batchSize)
                # 随机取样器
                randomSampler = lambda subset: torch.utils.data.sampler.RandomSampler(
                    subset, 
                    num_samples = decreaseInterval*batchSize_int, 
                    replacement = True
                )
                train_random_loaders_splited = [torch.utils.data.DataLoader(
                    dataset = subset,
                    batch_size = batchSize_int, 
                    sampler = randomSampler(subset),
                ) for subset in train_dataset_subset]
                randomIters = [iter(loader) for loader in train_random_loaders_splited]

            # 诚实节点更新
            for node in range(nodeSize):
                # 读取数据
                material, targets = next(randomIters[node])
                material, targets = material.to(device), targets.to(device)
                
                # 随机梯度
                # --------------------
                # 预测
                outputs = model(material)
                loss = loss_func(outputs, targets)
                # 反向传播
                model.zero_grad()
                loss.backward()
                # 加上weight_decay
                for para in model.parameters():
                    para.grad.data.add_(weight_decay, para)

                # censor
                # --------------------
                # 理论确定的常数
                w = 1/60
                tau = (w*sum(gradientRecord[node])/D) / (nodeSize**2)
                # 梯度变化
                diff = sum([((oldGrad-para.grad.data).norm()**2).item()
                            for (oldGrad, para) in zip(messages[node], model.parameters())])
                if diff > tau:
                    for (para, grad) in zip(model.parameters(), messages[node]):
                        grad.data.copy_(para.grad.data)
                    gradientNorm = sum([(para.grad.data.norm()**2).item() 
                                         for para in model.parameters()
                                        ])
                    gradientRecord[node].append(gradientNorm)
                    communicationCost += 1
                else:
                    gradientRecord[node].append(gradientRecord[node][-1])
                gradientRecord[node].pop(0)
                # --------------------

                # 记录准确率和损失
                _, predicted = torch.max(outputs.data, dim=1)
                trainAccAvg += (predicted == targets).sum().item()
                trainLossAvg += loss.item()
                totalSample += len(targets)
                
            # 更新
            for message in messages:
                for para, grad in zip(model.parameters(), message):
                    para.data.add_(-gamma / nodeSize, grad)
                   
            # 更新CSGD参数
            if iteration != 0 and iteration % decreaseInterval == 0:
                batchSize *= eta1
        
        # 记录信息
        communicationPath.append(communicationCost)
        
        trainLoss, trainAcc = trainLossAvg / totalSample, trainAccAvg / totalSample
        valLoss, valAcc = calculateAccuracy(model, validate_loader, device)

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAcc)
        valLossPath.append(valLoss)
        valAccPath.append(valAcc)
        
        recorder.report(r+1, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath)
    return model, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath

# %% [markdown]
# # 训练函数

# %%
def train(model, loss_func, optimizer, trainloader, device, weight_decay):
    """
    train model using loss_fn and optimizer in an epoch.
    model: CNN networks
    train_loader: a Dataloader object with training data
    loss_func: loss function
    device: train on cpu or gpu device
    """
    model.train()
    
    trainAccuracy = 0
    trainLoss = 0
    total = 0
    
    for i, (*material, targets) in enumerate(trainloader):
        if isinstance(material, torch.Tensor):
            material = material.to(device)
        else:
            material = [m.to(device) for m in material]
        
        targets = targets.to(device)

        # forward
        outputs = model(*material)
        
        loss = loss_func(outputs, targets)
        trainLoss += loss.item()

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # AdamW - https://zhuanlan.zhihu.com/p/38945390
        for group in optimizer.param_groups:
            for param in group['params']:
                param.data = param.data.add(-weight_decay * group['lr'], param.data)

        # return the maximum value of each row of the input tensor in the 
        # given dimension dim, the second return vale is the index location
        # of each maxium value found(argmax)
        _, predicted = torch.max(outputs.data, dim=1)
        trainAccuracy += (predicted == targets).sum().item()
        
        total += len(targets)
    trainAccuracy /= total
    trainLoss /= total
    return trainLoss, trainAccuracy


# %%
def validate(model, loss_func, validateloader, device):
    # evaluate the model
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        # =============================================================
        valAccuracy = 0
        valLoss = 0
        total = 0
        
        for i, (*material, targets) in enumerate(trainloader):
            if isinstance(material, torch.Tensor):
                material = material.to(device)
            else:
                material = [m.to(device) for m in material]

            targets = targets.to(device)
            
            outputs = model(*material)
            
            loss = loss_func(outputs, targets)
            valLoss += loss.item()
            
            # return the maximum value of each row of the input tensor in the 
            # given dimension dim, the second return vale is the index location
            # of each maxium value found(argmax)
            _, predicted = torch.max(outputs.data, dim=1)
            valAccuracy += (predicted == targets).sum().item()
            
            total += len(targets)
        valAccuracy /= total
        valLoss /= total
    return valLoss, valAccuracy


# %%
def test(model, testloader, classname=None, name='default'):
    # evaluate the model
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        result = []
        test_cnt = 0
        for i, (*material, targets) in enumerate(testloader):
            if isinstance(material, torch.Tensor):
                material = material.to(device)
            else:
                material = [m.to(device) for m in material]

            targets = targets.to(device)

            outputs = model(*material)

            _, predicted = torch.max(outputs.data, dim=1)

            result.extend(predicted)
            test_cnt += len(targets)

    if classname != None:
        result = [classname[i] for i in result]

    log('共预测{}个数据'.format(test_cnt))
    df_predict = pd.DataFrame({'id': list(range(1, len(result)+1)), 'polarity': result})
    df_predict.to_csv('{}.csv'.format(name), index=False)
    log('预测完成')
    


# %%
def showCurve(list_trainLoss, list_trainAccuracy, list_valLoss, list_valAccuracy):
    xAxis = list(range(len(list_trainLoss)))
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(xAxis, list_trainLoss, label='train')
    axs[0].plot(xAxis, list_valLoss, label='validation')
    axs[0].set_title('Loss')

    axs[1].plot(xAxis, list_trainAccuracy, label='train')
    axs[1].plot(xAxis, list_valAccuracy, label='validation')
    axs[1].set_title('Accuracy')

    for ax in axs:
        ax.axis()
        ax.set_xlabel('epoch')
        ax.set_ylabel('{}'.format(ax.get_title()))
        ax.legend()
    fig.set_size_inches((8, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.show()

# %% [markdown]
# # 驱动函数

# %%
@releaseCUDA
def run(optimizer, config, device='cpu', recordInFile=True, markOnTitle=''):
    print('正在初始化...')

    # 模型
    model = modelFactory(fixSeed=config['fixSeed'], SEED=config['SEED'])
    model = model.to(device)

    # e.g. CIFAR10_Resnet18_CSGD
    title = '{}_{}_{}'.format(
        dataSetConfig['name'],
        model.__class__.__name__, 
        optimizer.__name__
    )
    
    if markOnTitle != '':
        title += '_' + markOnTitle
        
    if config['fixSeed']:
        fixSeed(SEED=config['SEED'])        
    
    # 打印运行信息
    print('[提交任务] ' + title)
    print('[运行信息]')
    print('[网络属性]   name={} parameters number={}'.format(model.__class__.__name__, getPara(model)))
    print('[优化方法]   name={}'.format(optimizer.__name__))
    print('[数据集属性] name={} trainSize={} validationSize={}'.format(dataSetConfig['name'], len(train_dataset), len(validate_dataset)))
    print('[优化器设置] gamma={} weight_decay={}'.format(config['gamma'], config['weight_decay']))
    print('[节点个数]   nodeSize={}'.format(config['nodeSize']))
    print('[运行次数]   rounds={}, displayInterval={}'.format(config['rounds'], config['displayInterval']))
    print('[torch设置]  device={}, SEED={}, fixSeed={}'.format(device, config['SEED'], config['fixSeed']))
    print('-------------------------------------------')
    
    try:
        # 开始运行
        log('提交任务')
        res = optimizer(model, device=device, **config)
        [*model, communicationPath, trainLossPath, trainAccPath, valLossPath, valAccPath] = res

        record = {
            **dataSetConfig,
            **{
                key:(config[key].__name__ if hasattr(config[key], '__call__') else config[key]) 
                for key in config
            },
            'gamma': config['gamma'],
            'weight_decay': config['weight_decay'],
            'nodeSize': config['nodeSize'],
            'rounds': config['rounds'],
            'displayInterval': config['displayInterval'],
            'communicationPath': communicationPath,
            'trainLossPath': trainLossPath, 
            'trainAccPath': trainAccPath, 
            'valLossPath': valLossPath, 
            'valAccPath': valAccPath, 
        }

        if recordInFile:
            with open(CACHE_DIR + title, 'wb') as f:
                pickle.dump(record, f)

    except Exception as e:
        traceback.print_exc()
