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

"""Conversion functions for unsupported ONNX nodes."""

import math
import keras
import numpy as np
from tao_byom.utils.convert_utils import ensure_tf_type, is_numpy
from convnext.erf import ERFLayer


def convert_erf(node, params, layers, lambda_func, node_name, keras_name):
    """Convert ERF.

    Tips for writing a conversion functions:
        * Inputs to the current node can be accessed by `layers[node.input[0]]`.
        * Params include node's attribtues necessary to instantiate a layer
        * Outputs to the current node be assigned to `layers[node_name]`
        * Any custom/lambda layers used during conversion must be passed to `lambda_func[node_name]`
    Reference: https://github.com/gmalivenko/onnx2keras/tree/master/onnx2keras

    Args:
        node: current operation node
        params: operation attributes
        layers: available keras layers
        lambda_func: function for keras Lambda layer
        node_name: resulting layer name

    Returns:
        None
    """
    # Input can be either be numpy (directly from ONNX node)
    # or TF Tensor (from previous keras layer)
    if is_numpy(layers[node.input[0]]):
        # input is a numpy (static variable) and does not requires any gradient computation
        input_0 = layers[node.input[0]]

        # output is passed to `layers[node_name]`
        layers[node_name] = np.vectorize(math.erf)(input_0)
    else:
        # confirm that input is actually a TF tensor
        input_0 = ensure_tf_type(layers[node.input[0]], name=f"{keras_name}_const")

        # instantitate custom Keras Layer that we implemented
        lambda_layer = ERFLayer(name=keras_name)

        # output is passed to `layers[node_name]`
        layers[node_name] = lambda_layer(input_0)

        # Custom layers should be passed to `lambda_func[node_name]`
        lambda_func[keras_name] = ERFLayer


def convert_reduce_prod(node, params, layers, lambda_func, node_name, keras_name):
    """Convert ReduceProd. Example of using keras backend + lambda layer

    Tips for writing a conversion functions:
        * Inputs to the current node can be accessed by `layers[node.input[0]]`.
        * Params include node's attribtues necessary to instantiate a layer
        * Outputs to the current node be assigned to `layers[node_name]`
        * Any custom/lambda layers used during conversion must be passed to `lambda_func[node_name]`
    Reference: https://github.com/gmalivenko/onnx2keras/tree/master/onnx2keras

    Args:
        node: current operation node
        params: operation attributes
        layers: available keras layers
        lambda_func: function for keras Lambda layer
        node_name: resulting layer name

    Returns:
        None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce_prod layer.')
    axis = params.get("axes", -1)
    keepdims = params.get("keepdims", 1)

    # Input can be either be numpy (directly from ONNX node)
    # or TF Tensor (from previous keras layer)
    if is_numpy(layers[node.input[0]]):
        input_0 = layers[node.input[0]]
        layers[node_name] = np.prod(input_0, axis=axis, keepdims=(keepdims == 1))
    else:
        # confirm that input is actually a TF tensor
        input_0 = ensure_tf_type(layers[node.input[0]], name=f"{keras_name}_const")

        # Lambda layer with an inner function is an easy way to building custom layer
        # without writing a separate custom class.
        # Note that there may be issues with deserializing
        def target_layer(x, axis=axis, keepdims=keepdims):
            # we must include external packages in inner function to be able
            # to load back lambda layers
            import keras.backend as K
            return K.prod(x, keepdims=(keepdims == 1), axis=axis)

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)

        # output is passed to `layers[node_name]`
        layers[node_name] = lambda_layer(input_0)

        # Lambda layers should be passed to `lambda_func[node_name]`
        lambda_func[keras_name] = target_layer
