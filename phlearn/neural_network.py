import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200, random_scale=1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_func = sigmoid
        self.learning_rate = self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_scale = random_scale

        self.params = {"w": None, "bias": None}

        pass

    def fit(self, X, y):

        L = len(self.hidden_layer_sizes)

        output_layer_size = y.shape[1]
        input_layer_size = X.shape[1]

        # 初始化w
        w = [None for i in range(len(self.hidden_layer_sizes) + 1)]
        for l in range(L + 1):
            if l == 0:
                w[l] = np.random.rand(input_layer_size, self.hidden_layer_sizes[0]) * self.random_scale
            elif l == L:
                w[l] = np.random.rand(self.hidden_layer_sizes[-1], output_layer_size) * self.random_scale
            else:
                w[l] = np.random.rand(self.hidden_layer_sizes[l - 1], self.hidden_layer_sizes[l]) * self.random_scale

        # 初始化bias
        bias = [np.random.rand(self.hidden_layer_sizes[i], 1) * self.random_scale for i in range(L)]
        # 输出层bias
        bias.append(np.random.rand(output_layer_size, 1) * self.random_scale)

        self.params['w'] = w
        self.params['bias'] = bias

        i = 0
        while i < self.max_iter:

            print('iter {0}'.format(i))
            self._do_one_iteration(X, y)

            # debug
            # 数据总误差
            E = 0
            for k, v in enumerate(X):
                output_y = self._forward_pass(X[k])[-1]
                Ek = np.sum(np.square(output_y - y[k]))
                E += Ek
            E /= len(X)
            print('E : {0}'.format(E))

            i += 1

    # 对数据集执行一次迭代
    def _do_one_iteration(self, X, y):

        for k, v in enumerate(X):
            # 前向传播 计算computed_y
            computed_y = self._forward_pass(X[k])
            self._backward_pass(X[k], y[k], computed_y)

    # 对指定的数据执行前向传播
    def _forward_pass(self, x):

        L = len(self.hidden_layer_sizes)

        # 前向传播 计算computed_y
        computed_y = [None for i in range(L + 1)]
        computed_y[0] = self.compute_f(self.params['w'][0], x, self.params['bias'][0])
        for i in range(1, L + 1):
            computed_y[i] = self.compute_f(self.params['w'][i], computed_y[i - 1], self.params['bias'][i])

        return computed_y

    # 后向传播
    def _backward_pass(self, x, y, computed_y):

        L = n_hidden_layers = len(self.hidden_layer_sizes)
        w = self.params['w']
        bias = self.params['bias']

        output_layer_size = len(y)
        input_layer_size = len(x)

        # 计算delta矩阵
        layer_delta = [None for i in range(n_hidden_layers + 1)]
        layer_delta[-1] = computed_y[-1] * (computed_y[-1] - y) * (1 - computed_y[-1])
        for l in range(L - 1, -1, -1):
            K = len(bias[l])
            layer_delta[l] = np.zeros((K,))
            for k in range(K):
                layer_delta[l][k] = np.sum(layer_delta[l + 1] * w[l + 1][k, :]) * \
                                    computed_y[l][k] * (1 - computed_y[l][k])

        # 计算delta_w矩阵
        # delta_w = delta * y
        delta_w = [np.zeros(w[i].shape) for i in range(len(w))]
        # 最后一层隐层与输出层之间的delta_w
        for i in range(self.hidden_layer_sizes[-1]):
            for j in range(output_layer_size):
                delta_w[L][i][j] = layer_delta[L][j] * computed_y[L - 1][i]
        # 迭代
        for l in range(L - 1, 0, -1):
            for i in range(self.hidden_layer_sizes[l - 1]):
                for j in range(self.hidden_layer_sizes[l]):
                    delta_w[l][i][j] = layer_delta[l][j] * computed_y[l - 1][i]
        # 第一层隐层与输入层之间的delta_w
        for i in range(input_layer_size):
            for j in range(self.hidden_layer_sizes[0]):
                delta_w[0][i][j] = layer_delta[0][j] * x[i]

        # 计算delta_bias矩阵
        # delta_bias = delta * (-1)
        delta_bias = [np.zeros(bias[i].shape) for i in range(len(bias))]
        # 最后一层隐层与输出层之间的delta_bias
        for j in range(output_layer_size):
            delta_bias[L][j] = layer_delta[L][j] * (-1)
        # 迭代
        for l in range(L - 1, -1, -1):
            for j in range(self.hidden_layer_sizes[l]):
                delta_bias[l][j] = layer_delta[l][j] * (-1)

        # 更新参数
        # v = v - eta * delta_v
        for i in range(len(w)):
            w[i] += -self.learning_rate * delta_w[i]
            bias[i] += -self.learning_rate * delta_bias[i]

    def predict(self, X):

        result = np.zeros((X.shape[0]), dtype=np.int32)

        for i, x in enumerate(X):
            output_y = self._forward_pass(X[i])[-1]

            result[i] = np.where(output_y == np.max(output_y))[0][0]

        return result

    def compute_f(self, w, y_forward, bias):
        n, m = w.shape
        y = np.zeros((m,))
        for i in range(m):
            y[i] = self.activation_func(np.sum(w[:, i] * y_forward) - bias[i])

        # print(y[0])
        return y
