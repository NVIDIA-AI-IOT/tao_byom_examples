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

import numpy as np
from mobilenetv3.hard_sigmoid import HardSigmoidLayer
from tao_byom.utils.convert_utils import ensure_tf_type, is_numpy


def convert_hard_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
    """Convert HardSigmoid activation layer.

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
        node_name: internal converter name
        keras_name: resulting layer name

    Returns:
        None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')
    alpha = params.get('alpha', 0.2)
    beta = params.get('beta', 0.5)

    # Input can be either be numpy (directly from ONNX node)
    # or TF Tensor (from previous keras layer)
    if is_numpy(layers[node.input[0]]):
        # input is a numpy (static variable) and does not requires any gradient computation
        input_0 = layers[node.input[0]]

        # output is passed to `layers[node_name]`
        layers[node_name] = np.clip(input_0 * alpha + beta, 0, 1)
    else:
        # confirm that input is actually a TF tensor
        input_0 = ensure_tf_type(layers[node.input[0]], name=f"{keras_name}_const")

        # instantitate custom Keras Layer that we implemented
        hard_sigmoid = HardSigmoidLayer(alpha, beta, name=keras_name)

        # output is passed to `layers[node_name]`
        layers[node_name] = hard_sigmoid(input_0)

        # Custom layers should be passed to `lambda_func[node_name]`
        lambda_func[node_name] = HardSigmoidLayer
