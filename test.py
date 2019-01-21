import math


def sigmoid(x):
    print( math.exp(-x) +1 )
    return 1 / (1 + math.exp(-x))


print(sigmoid(293))
