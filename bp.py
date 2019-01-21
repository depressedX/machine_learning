# -*- coding: utf-8 -*-

from sklearn import datasets
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier as MLPClassifierLib
from phlearn.neural_network import MLPClassifier

iris = datasets.load_digits()  # 导入数据集
X = iris.data  # 获得其特征向量
y_target = iris.target  # 获得样本label


# 数据预处理
y = np.zeros((y_target.shape[0], 10))
for index, target in enumerate(y_target):
    y[index] = [1 if target == i else 0 for i in range(10)]

clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, learning_rate_init=0.4, random_scale=.000001)
# clf = MLPClassifierLib(hidden_layer_sizes=(100,100,100),activation='logistic')

sX = X[0:100]
sy = y[0:100]

clf.fit(sX, sy)

print(y_target[100:150] == clf.predict(X[100:150]))

# plt.figure("MLP")
#
# for i in range(36):
#     plt.subplot(6,6,i+1)
#     plt.imshow(X[i].reshape((8,8)),cmap='gray')
#     plt.axis('off') # 关掉坐标轴为 off
#
#
#
# plt.show()
