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


class ERFLayer(keras.layers.Layer):
    """Custom Keras ERF Layer.

    ERF in ONNX computes error function of the given input tensor elementwise.
    There's no Keras operations for ERF so we need to use `tf.math`.
    """
    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)

    def call(self, inputs):
        """ERF"""
        return tf.math.erf(inputs)

    def get_config(self):
        """Keras layer get config."""
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
