import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 4e-4   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 6e-5  # L2正则化
batchsize = 64

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    # print(mnist.num_feat)
    # print(mnist.mean_X.size, mnist.std_X.size)
    nodes = [StdScaler(mnist.mean_X, mnist.std_X),
             Linear(784, 392),
             Dropout(),
             Linear(392, 196),
             BatchNorm(196),
             relu(),

             Linear(196, 98),
             Dropout(),
             relu(),

             Linear(98, 49),
             Dropout(),
             relu(),

             Linear(49, mnist.num_class),
             Softmax(),
             CrossEntropyLoss(Y)]
    graph=Graph(nodes)
    return graph
    # TODO: YOUR CODE HERE

# def buildGraph(Y):
#     """
#     建图
#     @param Y: n 样本的label
#     @return: Graph类的实例，建好的图
#     """
#     nodes = [StdScaler(mnist.mean_X, mnist.std_X), Linear(mnist.num_feat, mnist.num_class), LogSoftmax(), NLLLoss(Y)]
#     graph=Graph(nodes)
#     return graph
