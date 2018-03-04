import warnings
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='warn')
warnings.filterwarnings('error')


def get_batch(source_dict, size, skip_count):
    return {k: source_dict[k][skip_count:size + skip_count] for k in source_dict.keys()}


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def to_output(source_dict):
    labels = source_dict['labels']
    y = np.zeros((len(labels), 10))

    for j, label in enumerate(labels):
        y_vector = np.zeros(10)
        y_vector[label] = 1
        y[j] = y_vector

    return y


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    try:
        return 1 / (1 + np.exp(-x))
    except Warning as e:
        raise e


def relu(x, deriv=False):
    if (deriv == True):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    x[x < 0] = 0
    return x


batch_size = 64
epoch_count = 50

E = 0.3
alpha = 0.00001

# np.random.seed(2)

synapse_0 = 2 * np.random.random((3072, 1024)) - 1
synapse_1 = 2 * np.random.random((1024, 10)) - 1

prev_synapse_0_delta = np.zeros((3072, 1024))
prev_synapse_1_delta = np.zeros((1024, 10))

# errors = np.zeros((epoch_count, 10000));
errors = []
synapse_0_delta_batch = np.zeros((3072, 1024))
synapse_1_delta_batch = np.zeros((1024, 10))

activation_func = nonlin

for i in xrange(epoch_count):

    # for x in range(1, 6):
    #     batch = unpickle("dataset/data_batch_"+ x)

    dataset_1 = unpickle("dataset/data_batch_1")
    for batch_number in range(0, 10000 / batch_size):

        mini_batch = get_batch(dataset_1, batch_size, batch_number * batch_size)
        y = to_output(mini_batch)  # batch_size x 10

        for j, X in enumerate(mini_batch['data']):
            y_local = y[j]  # 1x10
            l0 = np.array(X)[np.newaxis]/float(255)  # 1x3072

            l1 = activation_func(np.dot(l0, synapse_0))  # 1x3072 * 3072x1024 = 1x1024
            l2 = activation_func(np.dot(l1, synapse_1))  # 1x1024 * 1024x10 = 1x10

            l2_error = y_local - l2  # 1x10 - 1x10

            error = np.mean(np.abs(l2_error))
            # errors[i, j + batch_number * batch_size] = error
            errors.append(error)
            print "Epoch " + str(i).zfill(2) + ". Batch " + str(batch_number).zfill(3) + ". Picture " + str(j).zfill(2) + ". Error = " + str(error) + ". Average error = " + str(np.mean(errors))

            l2_delta = l2_error * activation_func(l2, deriv=True)  # 1x10 * 1x10

            l1_error = l2_delta.dot(synapse_1.T)  # 1x10 * 10x1024 = 1x1024

            l1_delta = l1_error * activation_func(l1, deriv=True)  # 1x1024 * 1x1024

            grad_1 = l1.T.dot(l2_delta)  # 1024x1 * 1x10 = 1024x10
            grad_0 = l0.T.dot(l1_delta)  # 3072x1 * 1x1024 = 3072x1024

            synapse_1_delta_batch += E * grad_1 + alpha * prev_synapse_1_delta
            synapse_0_delta_batch += E * grad_0 + alpha * prev_synapse_0_delta

        synapse_1 += synapse_1_delta_batch
        synapse_0 += synapse_0_delta_batch

        prev_synapse_1_delta = synapse_1_delta_batch
        prev_synapse_0_delta = synapse_0_delta_batch

        synapse_1_delta_batch = np.zeros((1024, 10))
        synapse_0_delta_batch = np.zeros((3072, 1024))

    plt.plot(errors)
    plt.show()
