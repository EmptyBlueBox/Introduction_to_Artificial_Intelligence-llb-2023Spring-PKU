import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 1  # 学习率
wd = 5e-4  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d
    @param bias: 1
    @return: wx+b
    """
    return np.matmul(X, weight) + bias


def sigmoid_one_element(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1.0)


def sigmoid(x):
    return np.array(list(map(sigmoid_one_element, x)))


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: n*d 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: d
    @param bias: 1
    @param Y: n 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: n 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: 1 由交叉熵损失函数计算得到
        weight: d 更新后的weight参数
        bias: 1 更新后的bias参数
    """
    haty = predict(X, weight, bias)  # 预测值
    loss = -np.mean(np.log(sigmoid(Y * haty) + 1e-6))  # 交叉熵损失函数
    Jw = -np.mean(((1.0 - sigmoid(Y * haty)) * Y).reshape(
        X.shape[0], -1) * X, axis=0) + 2 * wd * weight  # 损失函数对weight的偏导
    Jb = -np.mean((1.0 - sigmoid(Y * haty)) * Y)# 损失函数对bias的偏导
    weight = weight - lr * Jw  # 更新weight
    bias = bias - lr * Jb  # 更新bias
    return haty, loss, weight, bias
