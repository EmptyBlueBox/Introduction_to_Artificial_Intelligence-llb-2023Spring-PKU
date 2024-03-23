from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 16     # 树的数量
ratio_data = 0.5   # 采样的数据比例
ratio_feat = 0.45 # 采样的特征比例
hyperparams = {"depth":60, "purity_bound":0.9, "gainfunc":negginiDA} # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    forest=[]
    for _ in range(num_tree):
        rand_data_index = np.random.permutation(np.arange(X.shape[0]))[0:int(X.shape[0]*ratio_data)]
        rand_data = X[rand_data_index]
        rand_Y = Y[rand_data_index]
        rand_unused_feat_index = list(np.random.permutation(np.arange(X.shape[1]))[0:int(X.shape[1]*ratio_feat)])
        forest.append(buildTree(rand_data, rand_Y, rand_unused_feat_index, **hyperparams))
    return forest

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
