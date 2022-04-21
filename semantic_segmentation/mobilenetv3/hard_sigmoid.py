# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Custom Keras layer implementation."""

import keras
import tensorflow as tf


class HardSigmoidLayer(keras.layers.Layer):
    """Custom Keras HardSigmoid Layer.

    HardSigmoid in ONNX is computed as y = max(0, min(1, alpha * x + beta)).
    Keras does not allow set values for alpha and beta.
    Reference: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/activations/hard_sigmoid.

    Attributes:
        alpha: Value of alpha
        beta: Value of beta
    """
    def __init__(self, alpha=0.2, beta=0.5, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        """Hard Sigmoid"""
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid
        return tf.clip_by_value(inputs * self.alpha + self.beta, 0, 1)

    def get_config(self):
        """Keras layer get config."""
        config = {"alpha": self.alpha,
                  "beta": self.beta}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
