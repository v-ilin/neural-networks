# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

E = 0.7
alpha = 0.3

#np.random.seed(1)

syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

prev_syn0_delta = np.zeros((3, 4))

prev_syn1_delta = np.zeros((4, 1))

epoch_count = 60000

errors = np.zeros(epoch_count);

for i in xrange(epoch_count):

    l0 = X # 4x3
    l1 = nonlin(np.dot(l0, syn0)) # 4x3 * 3x4 = 4x4
    l2 = nonlin(np.dot(l1, syn1)) # 4x4 * 4x1 = 4x1

    l2_error = y - l2 # 4x1 - 4x1

    error = np.mean(np.abs(l2_error))
    errors[i] = error

    l2_delta = l2_error * nonlin(l2, deriv=True) # 4x1 * 4x1 = 4x1

    l1_error = l2_delta.dot(syn1.T) # 4x1 * 1x4 = 4x4

    l1_delta = l1_error * nonlin(l1, deriv=True) # 4x4 * 4x4

    grad1 = l1.T.dot(l2_delta) # 4x4 * 4x1 = 4x1
    grad2 = l0.T.dot(l1_delta) # 3x4 * 4x4 = 3x4

    syn0_delta = E * grad2 + alpha * prev_syn0_delta
    syn1_delta = E * grad1 + alpha * prev_syn1_delta
    prev_syn1_delta = syn1_delta

    syn1 += syn1_delta
    syn0 += syn0_delta

plt.plot(errors)
plt.show()