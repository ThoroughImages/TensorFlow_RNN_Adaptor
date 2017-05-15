# TensorFlow_RNN_Adaptor
Easily Transfer TensorFlow 0.12 RNN Model File to TensorFlow 1.0.

## Overview
The RNN definition for TF 0.12 is different from 1.0: tf.nn. * -> tf.contrib.rnn. *

The model is not compatible, in other words, once we trained a model on TF 0.12, it is hard to restore it on TF 1.0.

By using this adapter, you can transfer TF 0.12 RNN model into TF 1.0.

## Usage
`tf_0.12_rnn_definition.py`  RNN definition of TF 0.12.

`tf_1.0_rnn_definition.py`  RNN definition of TF 1.0.

`adaptor.py`  Transfer TF 0.12 RNN model into TF 1.0.

## Quick Tutorial

[1] Train and save your model in TF 0.12 environment with TF 0.12 RNN definition.

[2] Run adaptor (in TF 0.12 environment along with TF 0.12 RNN definition) to transfer the TF 0.12 RNN model to TF 1.0.

[3] Restore your model in TF 1.0 environment with TF 1.0 RNN definition.

## Related Links
[1] https://github.com/tensorflow/tensorflow/issues/7664

[2] https://github.com/tensorflow/tensorflow/issues/2768
