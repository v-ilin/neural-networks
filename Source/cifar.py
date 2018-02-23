def get_batch(source_dict, size, skip_count):
    return {k: source_dict[k][skip_count:size + skip_count] for k in source_dict.keys()}

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

batch_size = 50

batch1 = unpickle("dataset/data_batch_1")