import warnings
import datetime
import numpy as np

np.seterr(all='warn')
warnings.filterwarnings('error')

biases_count = 100
mini_batch_size = 64
epoch_count = 50

E = 0.5
alpha = 0.25

# np.random.seed(2)

synapse_0 = 2 * np.random.random((3072 + biases_count, 1024)) - 1
synapse_1 = 2 * np.random.random((1024 + biases_count, 10)) - 1

prev_synapse_0_delta = np.zeros((3072 + biases_count, 1024))
prev_synapse_1_delta = np.zeros((1024 + biases_count, 10))

errors = []

synapse_0_delta_batch = np.zeros((3072 + biases_count, 1024))
synapse_1_delta_batch = np.zeros((1024 + biases_count, 10))


batch_size = 10000

date = datetime.datetime.now().strftime("%m.%d.%y-%H.%M.%S")


def get_mini_batch_indexes(batch_indexes, size, skip_count):
    return batch_indexes[skip_count:skip_count + size]


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def get_mini_batch_y(source_dict, mini_batch_indexes):
    y_mini_batch = np.zeros((len(mini_batch_indexes), 10))

    for t, index in enumerate(mini_batch_indexes):
        y_vector = np.zeros((1, 10))
        y_vector[0][source_dict['labels'][index]] = 1
        y_mini_batch[t] = y_vector

    return y_mini_batch


def get_mini_batch_x(source_dict, mini_batch_indexes):
    x_mini_batch = np.zeros((len(mini_batch_indexes), 3072))

    for t, index in enumerate(mini_batch_indexes):
        x_mini_batch[t] = source_dict['data'][index]

    return x_mini_batch


def activation_func(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    try:
        return 1 / (1 + np.exp(-x))
    except Warning as e:
        dump_weights("error")
        raise e


def dump_weights(postfix):
    import os
    import cPickle

    directory_name = "e:/dumps/{}".format(date)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    synapse_0_filepath = "{}/synapse_0_{}.pkl".format(directory_name, postfix)
    synapse_1_filepath = "{}/synapse_1_{}.pkl".format(directory_name, postfix)

    with open(synapse_0_filepath, 'wb') as synapse_0_file:
        cPickle.dump(synapse_0, synapse_0_file)
        synapse_0_file.close()

    with open(synapse_1_filepath, 'wb') as synapse_1_file:
        cPickle.dump(synapse_1, synapse_1_file)
        synapse_1_file.close()


dump_weights("initial")

for i in xrange(epoch_count):

    for batch_number in range(1, 6):
        batch = unpickle("dataset/data_batch_{}".format(batch_number))
        batch_indexes = np.arange(batch['data'].shape[0])
        np.random.shuffle(batch_indexes)

        for mini_batch_number in range(0, batch_size / mini_batch_size):

            mini_batch_indexes = get_mini_batch_indexes(batch_indexes, mini_batch_size, mini_batch_number * mini_batch_size)
            x_mini_batch = get_mini_batch_x(batch, mini_batch_indexes)  # mini_batch_size x 3072
            y_mini_batch = get_mini_batch_y(batch, mini_batch_indexes)  # mini_batch_size x 10

            predictions = []

            for j, X in enumerate(x_mini_batch):
                y_local = y_mini_batch[j]  # 1x10
                l0_input = np.array(X)[np.newaxis] / float(255)  # 1x3072 , input normalization

                if biases_count != 0:
                    for _ in xrange(biases_count):
                        l0_input = np.append(l0_input, 1)
                    l0_input = np.array(l0_input)[np.newaxis]

                l0_sum = np.dot(l0_input, synapse_0)

                l1 = activation_func(l0_sum)  # 1x3072 * 3072x1024 = 1x1024

                if biases_count != 0:
                    for _ in xrange(biases_count):
                        l1 = np.append(l1, 1)
                    l1 = np.array(l1)[np.newaxis]

                l2 = activation_func(np.dot(l1, synapse_1))  # 1x1024 * 1024x10 = 1x10

                l2_error = y_local - l2  # 1x10 - 1x10

                error = np.mean(np.abs(l2_error))
                errors.append(error)

                l2_delta = l2_error * activation_func(l2, deriv=True)  # 1x10 * 1x10

                l1_error = l2_delta.dot(synapse_1.T)  # 1x10 * 10x1024 = 1x1024

                l1_delta = l1_error * activation_func(l1, deriv=True)  # 1x1024 * 1x1024

                grad_1 = l1.T.dot(l2_delta)  # 1024x1 * 1x10 = 1024x10
                grad_0 = l0_input.T.dot(l1_delta)  # 3072x1 * 1x1024 = 3072x1024

                synapse_1_delta_batch += E * grad_1 + alpha * prev_synapse_1_delta

                if biases_count != 0:
                    synapse_0_delta_batch += E * grad_0[:, :-biases_count] + alpha * prev_synapse_0_delta
                else:
                    synapse_0_delta_batch += E * grad_0 + alpha * prev_synapse_0_delta

                prediction = np.argmax(l2)
                if prediction == np.nonzero(y_local)[0]:
                    predictions.append(1)
                else:
                    predictions.append(0)

            synapse_1_delta_batch = synapse_1_delta_batch / mini_batch_size
            synapse_0_delta_batch = synapse_0_delta_batch / mini_batch_size

            synapse_1 += synapse_1_delta_batch
            synapse_0 += synapse_0_delta_batch

            prev_synapse_1_delta = synapse_1_delta_batch
            prev_synapse_0_delta = synapse_0_delta_batch

            synapse_1_delta_batch = np.zeros((1024 + biases_count, 10))
            synapse_0_delta_batch = np.zeros((3072 + biases_count, 1024))

            print "Epoch " + str(i).zfill(2) + ". " \
                  "Batch " + str(batch_number) + ". " \
                  "Mini-batch " + str(mini_batch_number).zfill(3) + ". " \
                  "Average mini-batch error = " + str(np.mean(errors[mini_batch_number * mini_batch_size:mini_batch_number * mini_batch_size + mini_batch_size])).zfill(3) + ". " \
                  "Success predictions = {}".format(np.count_nonzero(predictions))

        dump_weights("epoch={}batch={}".format(i, batch_number))

    dump_weights("epoch={}".format(i))

    # plt.plot(errors)
    # plt.show()
