import numpy as np
import random


class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        pass

    def split(self, X):
        size = len(X) // self.n_splits
        x_copy = np.array(X).copy()
        x_index = [i for i in range(len(X))]

        # 产生n_split个分组
        x_group = [[] for j in range(self.n_splits)]
        for i in range(self.n_splits):

            for j in range(size + (1 if i < len(X) % self.n_splits else 0)):
                # 随机从x_index中抽取j个元素 并从中删除
                selected_index = random.randint(0, len(x_index) - 1)
                x_group[i].append(x_index[selected_index])
                x_index.pop(selected_index)

        for i in range(self.n_splits):
            train_index = []
            test_index = x_group[i]
            for j in range(self.n_splits):
                if not j == i:
                    train_index += x_group[j]
            yield train_index, test_index


def cross_val_score(model, X, y):
    n_splits = 5

    kf = KFold(n_splits=n_splits)

    accurate_list = []

    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        model.fit(train_X, train_y)

        result_y = model.predict(test_X)
        accurate_list.append(len(test_y[test_y == result_y]) / len(test_y))

    return accurate_list
