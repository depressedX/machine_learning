# -*- coding: utf-8 -*-

from sklearn.cluster import Birch
from sklearn.cluster import KMeans as KMeansRefer
import math
import numpy as np
import matplotlib.pyplot as plt


def distance(x1, x2, dim):
    dist = 0
    for i in range(dim):
        dist += math.pow(x1[i] - x2[i], 2)
    dist = math.sqrt(dist)
    return dist


# 从y_set中选出距离最近 的点
def min_distance(x, y_set, dim):
    min_dist = math.inf
    min_index = -1
    for i in range(len(y_set)):
        dist = distance(x, y_set[i], dim)
        if dist < min_dist:
            min_index = i
            min_dist = dist
    return min_index, min_dist


# 集合中心
def set_center(x):
    c = np.zeros(x.shape[1:])
    for i in range(len(x)):
        c += x[i]
    c /= len(x)
    return c


class KMeans(object):
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit_predict(self, X):
        X = np.array(X)
        n = len(X)
        dim = X.shape[1]

        # 选取n_clusters个中心点
        cluster_centers = X[0:self.n_clusters].copy()
        X_cluster_index = np.zeros(n, dtype=np.uint8)

        convergent = False
        iter_time = 0
        while not convergent and iter_time < self.max_iter:
            convergent = True
            for i in range(n):
                min_index, min_dist = min_distance(X[i], cluster_centers, dim)
                if X_cluster_index[i] != min_index:
                    # 未收敛
                    convergent = False
                    X_cluster_index[i] = min_index

            # 更新中心点

            for i in range(self.n_clusters):
                sub_set = X[X_cluster_index == i]
                if not len(sub_set) == 0:
                    cluster_centers[i] = set_center(sub_set)

            iter_time += 1

        return X_cluster_index


X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1906, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.2521, 0.5735],
     [0.1007, 0.6318],
     [0.1067, 0.4326],  # amrk
     [0.1956, 0.4280]
     ]
X = np.array(X)

# 库函数 Kmeans聚类
clf = KMeansRefer(n_clusters=3)
y_pred = clf.fit_predict(X)

x = [n[0] for n in X]
y = [n[1] for n in X]

# Kmeans聚类
clf1 = KMeans(n_clusters=3)
y_pred1 = clf1.fit_predict(X)

# 可视化操作
plt.subplot(2, 1, 1)
plt.scatter(x, y, c=y_pred, marker='x')
plt.title("lib Kmeans-Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend(["Rank"])

plt.subplot(2, 1, 2)
plt.scatter(x, y, c=y_pred1, marker='x')
plt.title("my Kmeans-Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend(["Rank"])
plt.show()
