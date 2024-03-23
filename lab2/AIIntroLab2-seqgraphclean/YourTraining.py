import numpy as np
from numpy.random import randn
import mnist
import pickle
from util import setseed


setseed(0)


_save_path_header = 'model/mymodel_'
lr = 1e-1
wd = 5e-4
epoch=100


def predict(x, _weight, _bias):
    return np.matmul(x, _weight) + _bias


def sigmoid_one_element(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1.0)


def sigmoid(x):
    return np.array(list(map(sigmoid_one_element, x)))


def step(x, weight, bias, y):
    y_hat = predict(x, weight, bias)
    loss = -np.mean(np.log(sigmoid(y * y_hat) + 1e-5))
    jw = -np.mean(((1.0 - sigmoid(y * y_hat)) *
                  y).reshape(x.shape[0], -1) * x, axis=0) + 2 * wd * weight
    jb = -np.mean((1.0 - sigmoid(y * y_hat)) * y)
    weight = weight - lr * jw
    bias = bias - lr * jb
    return y_hat, loss, weight, bias


X = mnist.trn_X
Y = mnist.trn_Y


if __name__ == '__main__':
    avg_acc = 0.0
    for model in range(10): #训练10个模型
        best_train_acc = 0
        tmpY = 2 * (Y == model) - 1
        weight = randn(mnist.num_feat)
        bias = 0
        for i in range(1, epoch + 1):
            haty, loss, weight, bias = step(
                X, weight, bias, tmpY)
            acc = np.average(haty * tmpY>0)
            # print(haty)
            if acc > best_train_acc:
                best_train_acc = acc
                with open(_save_path_header + str(model) + '.npy', 'wb') as f:
                    pickle.dump((weight, bias), f)

    #     with open(_save_path_header + str(model) + '.npy', 'rb') as f:
    #         weight, bias = pickle.load(f)
    #     haty = predict(mnist.val_X, weight, bias)
    #     haty = (haty > 0)
    #     y = (mnist.val_Y == model)    
    #     print(
    #         f'confusion matrix: TP {np.sum((haty > 0) * (y > 0))} FN {np.sum((y > 0) * (haty <= 0))} FP {np.sum((y <= 0) * (haty > 0))} TN {np.sum((y <= 0) * (haty <= 0))}')
    #     valid_acc = np.average(haty == y)
    #     avg_acc += valid_acc
    #     print(f'valid acc {valid_acc:.4f}')
    # print(f'avg acc {avg_acc / 10:.4f}')
