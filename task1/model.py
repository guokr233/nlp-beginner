import numpy as np


class SoftmaxClassifier(object):
    def __init__(self, vocab_size, classes=5):
        # self.w = np.random.randn(vocab_size+1, classes)
        self.w = np.zeros((vocab_size+1, classes))
        # self.b = np.zeros((1, classes))

    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def forward(self, x):
        """
        x: (batch_size, vocab_size)
        w.T: (classes, vocab_size)
        b: (1, classes)
        :return (batch_size, vocab_size)
        """
        z = np.dot(x, self.w)
        return self.softmax(z)

    def predict(self, x):
        prob = self.forward(x)
        return np.argmax(prob, axis=1)

    def step(self, x, target, lr):
        batch_grad = self.batch_gradient(x, target)
        self.w += lr * batch_grad

    # 计算一个batch样本的平均梯度
    def batch_gradient(self, x, target):
        """
            x: (batch_size, vocab_size)
            target: (batch_size, classes)
        """
        pred = self.forward(x)  # (batch_size, classes)
        # target = one_hot(target)  # (batch_size, classes)
        return np.dot(x.T, target - pred) / x.shape[0]  # 梯度下降算法公式





