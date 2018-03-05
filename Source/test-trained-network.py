import numpy as np


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def activation_func(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def y_to_vector(y):
    y_vector = np.zeros((1, 10))
    y_vector[0][y] = 1
    return y_vector


synapse_0 = unpickle("e:/dumps/03.05.18-23.07.40/synapse_0_epoch=0batch=0mb=48.pkl")
synapse_1 = unpickle("e:/dumps/03.05.18-23.07.40/synapse_1_epoch=0batch=0mb=48.pkl")

test_batch = unpickle("dataset/test_batch")

errors = []
predictions = []

for i, X in enumerate(test_batch['data']):
    l0 = np.array(X)[np.newaxis] / float(255)  # 1x3072
    l1 = activation_func(np.dot(l0, synapse_0))  # 1x3072 * 3072x1024 = 1x1024
    l2 = activation_func(np.dot(l1, synapse_1))  # 1x1024 * 1024x10 = 1x10

    y = test_batch['labels'][i]
    l2_error = y_to_vector(y) - l2  # 1x10 - 1x10

    prediction = np.argmax(l2)
    if prediction == y:
        predictions.append(1)
    else:
        predictions.append(0)

    error = np.mean(np.abs(l2_error))
    errors.append(error)

    print "Image {}. Error = {}. Average error = {}".format(i, error, np.mean(errors))

print "Right predictions count = {} of 10000".format(np.count_nonzero(predictions))

