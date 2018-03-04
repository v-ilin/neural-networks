import cPickle

synapse_0_filepath = "dumps/synapse_0_03.04.18-18.32.55.pkl"
synapse_1_filepath = "dumps/synapse_1_03.04.18-18.32.55.pkl"


synapse_0_file = open(synapse_0_filepath, 'rb')
synapse_1_file = open(synapse_1_filepath, 'rb')

synapse_0 = cPickle.load(synapse_0_file)
synapse_1 = cPickle.load(synapse_1_file)