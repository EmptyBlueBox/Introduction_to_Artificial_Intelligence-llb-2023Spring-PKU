from typing import List
import math
import numpy as np
import numpy as np
from .Init import *


def shape(x):
    if isinstance(x, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(x)):
            ret += "_posinf"
        if np.any(np.isneginf(x)):
            ret += "_neginf"
        if np.any(np.isnan(x)):
            ret += "_nan"
        return f" {np.shape(X)} "
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        ret = "float"
        if np.any(np.isposinf(x)):
            ret += "_posinf"
        if np.any(np.isneginf(x)):
            ret += "_neginf"
        if np.any(np.isnan(x)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(x)}")


class Node(object):
    def __init__(self, name, *params):
        # 节点的梯度
        self.grad = []
        # 节点保存的临时数据
        self.cache = []
        # 节点的名字
        self.name = name
        # 用于Linear节点中存储weight和bias参数使用
        self.params = list(params)

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度
        '''
        pass

    def flush(self):
        # 初始化/刷新
        self.grad = []
        self.cache = []

    def forward(self, x, debug=False):
        '''
        正向传播
        '''
        if debug:
            print(self.name, shape(x))
        ret = self.cal(x)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret

    def eval(self):
        pass

    def train(self):
        pass


class relu(Node):
    # shape x: (*)
    # shape value: (*) relu(x)
    def __init__(self):
        super().__init__("relu")

    def cal(self, x):
        self.cache.append(x)
        return np.clip(x, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0)


class sigmoid(Node):
    # shape x: (*)
    # shape value: (*) sigmoid(x)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, x):
        # TODO: YOUR CODE HERE
        ret = 0.5 * (1.0 + np.tanh(0.5 * x))
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        return np.multiply(grad, np.multiply(self.cache[-1], 1 - self.cache[-1]))


class tanh(Node):
    # shape x: (*)
    # shape value: (*) sigmoid(x)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, x):
        ret = np.tanh(x)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        return np.multiply(grad, np.multiply(1 + self.cache[-1], 1 - self.cache[-1]))


class Linear(Node):
    # shape x: (*,d1)
    # shape weight: (d1, d2)
    # shape bias: (d2)
    # shape value: (*, d2) 
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        # TODO: YOUR CODE HERE
        weight = self.params[0]
        bias = self.params[1]
        self.cache.append(X)
        ret = X @ weight + bias
        return ret

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        self.grad.append(np.transpose(self.cache[-1]) @ grad)
        self.grad.append(np.sum(grad, axis=0))
        return grad @ np.transpose(self.params[0])


class StdScaler(Node):
    '''
    input shape (*)
    output shape (*)
    '''
    EPS = 1e-3

    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad / (self.std + self.EPS)


class BatchNorm(Node):
    '''
    input shape (*)
    output shape (*)
    '''
    EPS = 1e-3

    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.std is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1 - self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1 - self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad * self.params[0]) / (self.std + self.EPS)

    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout(Node):
    '''
    input shape (*)
    output shape (*)
    '''

    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0 <= p <= 1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*np.shape(X)) < self.p
            np.putmask(X, mask, 0)
            self.cache.append(mask)
        else:
            X = X * (1 / (1 - self.p))
        return X

    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            return grad
        else:
            return (1 / (1 - self.p)) * grad

    def eval(self):
        self.dropout = False

    def train(self):
        self.dropout = True


class Softmax(Node):
    # shape x: (*)
    # shape value: (*), softmax at dim 
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)


class LogSoftmax(Node):
    # shape x: (*)
    # shape value: (*), logsoftmax at dim 
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        # TODO: YOUR CODE HERE
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        lhj = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(lhj)
        ret = np.log(lhj + 1e-6)
        return ret

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        softmaxX = self.cache[-1]
        return grad - grad.sum(axis=self.dim, keepdims=True) * softmaxX


class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''

    # shape x: (*, d), y: (*)
    # shape value: number 
    # 输入：x: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret


class CrossEntropyLoss(Node):
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。建议先完成作业3第4题。
    '''

    # shape x: (*, d), y: (*)
    # shape value: number 
    # 输入：x: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        # TODO: YOUR CODE HERE
        # 提示，可以对照CrossEntropyLoss的cal
        self.cache.append(X)
        tmp = np.zeros(np.shape(X))
        tmp2 = np.ones(np.shape(self.y))
        tmp[np.arange(len(tmp)), self.y] = tmp2
        return -np.sum(np.multiply(tmp, np.log(X + 1e-6)))

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        # 提示，可以对照CrossEntropyLoss的backcal
        X = self.cache[-1]
        ret = np.zeros(np.shape(X))
        tmp = 1.0 / X[np.arange(len(X)), self.y]
        ret[np.arange(len(ret)), self.y] = -tmp
        return ret


class Reshape(Node):
    def __init__(self, out_shape):
        super().__init__("Reshape")
        self.out_shape = out_shape

    def cal(self, X):
        self.cache.append(X)
        return X.reshape(self.out_shape)

    def backcal(self, grad):
        return grad.reshape(np.shape(self.cache[-1]))


class Conv2d(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        din = in_channels * (kernel_size ** 2)
        weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) / np.sqrt(din)
        super().__init__("Conv2d", weight)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def cal(self, X):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """
        weight = self.params[0]
        self.cache.append(X)
        H1 = (np.shape(X)[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        W1 = (np.shape(X)[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = np.zeros((np.shape(X)[0], np.shape(weight)[0], H1, W1))
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                tmp1 = np.zeros((np.shape(X)[2] + 2 * self.padding, np.shape(X)[3] + 2 * self.padding))
                tmp1[self.padding: np.shape(tmp1)[0] - self.padding, self.padding: np.shape(tmp1)[1] - self.padding] = \
                X[i, j]
                for s1 in range(np.shape(weight)[0]):
                    for i1 in range(H1):
                        for j1 in range(W1):
                            i2 = i1 * self.stride
                            j2 = j1 * self.stride
                            out[i, s1, i1, j1] += np.sum(
                                weight[s1, j] * tmp1[i2: i2 + self.kernel_size, j2: j2 + self.kernel_size])
        return out

    def backcal(self, grad):
        X = self.cache[-1]
        weight = self.params[0]
        H1 = (np.shape(X)[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        W1 = (np.shape(X)[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
        weight_grad = np.zeros(np.shape(weight))
        X_grad = np.zeros(np.shape(X))
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                tmp1 = np.zeros((np.shape(X)[2] + 2 * self.padding, np.shape(X)[3] + 2 * self.padding))
                tmp_grad = np.zeros((np.shape(X)[2] + 2 * self.padding, np.shape(X)[3] + 2 * self.padding))
                tmp1[self.padding: np.shape(tmp1)[0] - self.padding, self.padding: np.shape(tmp1)[1] - self.padding] = \
                X[i, j]
                for s1 in range(np.shape(weight)[0]):
                    for i1 in range(H1):
                        for j1 in range(W1):
                            i2 = i1 * self.stride
                            j2 = j1 * self.stride
                            tmp_grad[i2: i2 + self.kernel_size, j2: j2 + self.kernel_size] += weight[s1, j] * grad[
                                i, s1, i1, j1]
                            weight_grad[s1, j] += tmp1[i2: i2 + self.kernel_size, j2: j2 + self.kernel_size] * grad[
                                i, s1, i1, j1]
                X_grad[i, j] = tmp_grad[self.padding: np.shape(tmp1)[0] - self.padding,
                               self.padding: np.shape(tmp1)[1] - self.padding]
        self.grad.append(weight_grad)
        return X_grad


class Conv2d_basic(Node):
    def __init__(self, in_channels, out_channels, kernel_size):
        din = in_channels * (kernel_size ** 2)
        weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) / np.sqrt(din)
        super().__init__("Conv2d_basic", weight)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def cal(self, X):
        weight = self.params[0]
        self.cache.append(X)
        H1 = np.shape(X)[2] - self.kernel_size + 1
        W1 = np.shape(X)[3] - self.kernel_size + 1
        out = np.zeros((np.shape(X)[0], np.shape(weight)[0], H1, W1))
        i0, j0 = np.meshgrid(range(self.kernel_size), range(self.kernel_size), indexing='ij')
        i1, j1 = np.meshgrid(range(H1), range(W1), indexing='ij')
        ii = i0.reshape(-1, 1) + i1.reshape(1, -1)
        jj = j0.reshape(-1, 1) + j1.reshape(1, -1)
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                selected = X[i, j, ii, jj]
                for s1 in range(np.shape(weight)[0]):
                    weights = weight[s1, j].reshape(1, -1)
                    out[i, j] += (weights @ selected).reshape(H1, W1)
        return out

    def backcal(self, grad):
        X = self.cache[-1]
        weight = self.params[0]
        H1 = np.shape(X)[2] - self.kernel_size + 1
        W1 = np.shape(X)[3] - self.kernel_size + 1
        weight_grad = np.zeros(np.shape(weight))
        X_grad = np.zeros(np.shape(X))
        grad = grad.reshape((np.shape(X)[0], np.shape(weight)[0], 1, -1))
        i0, j0 = np.meshgrid(range(self.kernel_size), range(self.kernel_size), indexing='ij')
        i1, j1 = np.meshgrid(range(H1), range(W1), indexing='ij')
        ii = i0.reshape(-1, 1) + i1.reshape(1, -1)
        jj = j0.reshape(-1, 1) + j1.reshape(1, -1)
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                selected = X[i, j, ii, jj]
                for s1 in range(np.shape(weight)[0]):
                    weights = weight[s1, j].reshape(1, -1)
                    X_grad[i, j, ii, jj] += np.transpose(weights) @ grad[i, j]
                    weight_grad[s1, j] += (grad[i, j] @ np.transpose(selected)).reshape((self.kernel_size, self.kernel_size))
        self.grad.append(weight_grad)
        return X_grad


class MaxPool2d(Node):
    def __init__(self, kernel_size):
        super().__init__("MaxPool2d")
        self.kernel_size = kernel_size

    def cal(self, X):
        self.cache.append(X)
        H1 = np.shape(X)[2] // self.kernel_size * self.kernel_size
        W1 = np.shape(X)[3] // self.kernel_size * self.kernel_size
        H2 = np.shape(X)[2] // self.kernel_size
        W2 = np.shape(X)[3] // self.kernel_size
        i0, j0 = np.meshgrid(range(self.kernel_size), range(self.kernel_size), indexing='ij')
        i1, j1 = np.meshgrid(range(0, H1, self.kernel_size), range(0, W1, self.kernel_size), indexing='ij')
        ii = i0.reshape(-1, 1) + i1.reshape(1, -1)
        jj = j0.reshape(-1, 1) + j1.reshape(1, -1)
        out = np.zeros((np.shape(X)[0], np.shape(X)[1], H2, W2))
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                selected = X[i, j, ii, jj]
                out[i, j] = np.max(selected, axis=0).reshape(H2, W2)
        return out

    def backcal(self, grad):
        X = self.cache[-1]
        X_grad = np.zeros(np.shape(X))
        H1 = np.shape(X)[2] // self.kernel_size * self.kernel_size
        W1 = np.shape(X)[3] // self.kernel_size * self.kernel_size
        i0, j0 = np.meshgrid(range(self.kernel_size), range(self.kernel_size), indexing='ij')
        i1, j1 = np.meshgrid(range(0, H1, self.kernel_size), range(0, W1, self.kernel_size), indexing='ij')
        ii = i0.reshape(-1, 1) + i1.reshape(1, -1)
        jj = j0.reshape(-1, 1) + j1.reshape(1, -1)
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                selected = X[i, j, ii, jj]
                id_1 = np.argmax(selected, axis=0)
                id_2 = np.arange(np.shape(selected)[1])
                x_lst = ii[id_1, id_2]
                y_lst = jj[id_1, id_2]
                X_grad[i, j, x_lst, y_lst] = grad[i, j].reshape(-1)
        return X_grad


class Residual(Node):
    def __init__(self, in_channels, out_channels, in_size, stride=1):
        super().__init__("Residual")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        out_size = in_size // stride
        self.left_conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # self.left_bn1 = BatchNorm(out_size)
        self.left_relu = relu()
        self.left_conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.left_bn2 = BatchNorm(out_size)
        self.shortcut_conv = None
        # self.shortcut_bn = None
        self.final_relu = relu()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm(out_size)

    def cal(self, X):
        left_X = self.left_conv1.cal(X)
        # left_X = self.left_bn1.cal(left_X)
        left_X = self.left_relu.cal(left_X)
        left_X = self.left_conv2.cal(left_X)
        # left_X = self.left_bn2.cal(left_X)
        if self.stride != 1 or self.in_channels != self.out_channels:
            sho_X = self.shortcut_conv.cal(X)
            # sho_X = self.shortcut_bn.cal(sho_X)
        else:
            sho_X = X
        res = left_X + sho_X
        res = self.final_relu.cal(res)
        return res

    def backcal(self, grad):
        grad_1 = self.final_relu.backcal(grad)
        # left_grad = self.left_bn2.backcal(grad_1)
        left_grad = self.left_conv2.backcal(grad_1)
        left_grad = self.left_relu.backcal(left_grad)
        # left_grad = self.left_bn1.backcal(left_grad)
        left_grad = self.left_conv1.backcal(left_grad)
        if self.stride != 1 or self.in_channels != self.out_channels:
            # sho_grad = self.shortcut_bn.backcal(grad_1)
            sho_grad = self.shortcut_conv.backcal(grad_1)
        else:
            sho_grad = grad_1
        return left_grad + sho_grad


