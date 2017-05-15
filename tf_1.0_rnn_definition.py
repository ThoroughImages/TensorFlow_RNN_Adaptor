import tensorflow as tf


with graph.as_default():
    # Placeholders
    sequence = tf.placeholder("float32", [None, max_length, frame_size])
    batch_size = tf.shape(sequence)[0]
    label = tf.placeholder("float32", [None, n_classes])

    # Define Weights and Bias for the Last FC Layer
    weight = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))

    # Define RNN
    next_batch = sequence
    seq_length = length(next_batch)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=True)
    _state = cell.zero_state(batch_size, tf.float32)  # Tensorflow LSTM cell requires 2 x n_hidden length (state & cell)
    output, states = tf.nn.dynamic_rnn(cell, next_batch, initial_state=_state, sequence_length=seq_length, scope="rnn")
    last = output[-1]
    pred = tf.nn.softmax(tf.matmul(last, weight) + bias)
