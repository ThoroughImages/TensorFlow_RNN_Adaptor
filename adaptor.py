import tensorflow as tf


# replace represents the mapping from TF 0.12 to TF 1.0 RNN definitions.
replace = dict()
for index in n_layers:
    replace['rnn/rnn/MultiRNNCell/Cell' + str(index) + '/BasicLSTMCell/Linear/Matrix'] = \
        'rnn/rnn/multi_rnn_cell/cell_' + str(index) + '/basic_lstm_cell/weights'


with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, "./4-64.ckpt")
    names_to_vars = {v.op.name: v for v in tf.all_variables()}
    for key in replace.keys():
        bias_var = names_to_vars[key]
        names_to_vars[replace[key]] = bias_var
        del names_to_vars[key]
    saver = tf.train.Saver(var_list=names_to_vars)
    saver.save(sess, 'new.ckpt')
