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


dumps_folder_name = "04.03.18-14.58.21"

synapse_0 = unpickle("e:/dumps/{}/synapse_0_epoch=99.pkl".format(dumps_folder_name))
synapse_1 = unpickle("e:/dumps/{}/synapse_1_epoch=99.pkl".format(dumps_folder_name))

test_batch = unpickle("dataset/test_batch")

biases_count = 1

errors = []
predictions = []

for i, X in enumerate(test_batch['data']):
    l0 = np.array(X)[np.newaxis] / float(255)  # 1x3072

    if biases_count != 0:
        for _ in xrange(biases_count):
            l0 = np.append(l0, 1)  # add one bias
        l0 = np.array(l0)[np.newaxis]

    l1 = activation_func(np.dot(l0, synapse_0))  # 1x3072 * 3072x50 = 1x50

    if biases_count != 0:
        for _ in xrange(biases_count):
            l1 = np.append(l1, 1)  # add one bias
        l1 = np.array(l1)[np.newaxis]

    l2 = activation_func(np.dot(l1, synapse_1))  # 1x50 * 50x10 = 1x10

    y = test_batch['labels'][i]
    l2_error = y_to_vector(y) - l2  # 1x10 - 1x10

    prediction = np.argmax(l2)
    if prediction == y:
        predictions.append(1)
    else:
        predictions.append(0)

    error = np.mean(np.abs(l2_error))
    errors.append(error)

    print "Image {}. Prediction = {}. Answer = {}. Error = {}. Average error = {}.".format(i, prediction, y, error, np.mean(errors))

print "Right predictions count = {} of 10000".format(np.count_nonzero(predictions))

