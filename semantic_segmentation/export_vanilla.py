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

"""Export UNet model from torch hub to ONNX."""

import os
import argparse
import onnx
import torch


def parse_command_line(args=None):
    """Parsing command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument('-x',
                        '--opset-version',
                        type=int,
                        required=False,
                        default=11,
                        help='Version of ONNX Opset. Default is set to 11/')
    parser.add_argument('--output_path',
                        type=str,
                        default=os.path.join(os.getcwd(), "onnx_models"),
                        help="Path to where the exported model is stored.",
                        required=False)
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=None,
                        help='input image height and width.')
    return parser.parse_args()


def main(args=None):
    """Run the training process."""
    args = parse_command_line(args)
    model_name = args.model_name

    print("Model Name: ", model_name)
    pretrained = 'scratch' not in model_name

    opset = args.opset_version

    os.makedirs(args.output_path, exist_ok=True)
    onnx_model_file = os.path.join(args.output_path, f"{model_name}.onnx")

    try:
        # https://github.com/pytorch/vision/issues/4156#issuecomment-894768539
        # Bug since torch==1.9.0
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=pretrained)
    except NotImplementedError:
        print(f"{model_name} is not supported by https://github.com/milesial/Pytorch-UNet")

    if args.shape is None:
        dummy_input = torch.randn(1, 3, 320, 320)
    elif len(args.shape) == 1:
        dummy_input = torch.randn(1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        dummy_input = torch.randn(1, 3, args.shape[0], args.shape[1])
    else:
        raise ValueError('invalid input shape')

    torch.onnx.export(model, dummy_input, onnx_model_file,  input_names=['input_1'],
                      verbose=False, training=2, export_params=True,
                      do_constant_folding=False,
                      opset_version=opset)

    onnx_model = onnx.load(onnx_model_file)
    # Check that the IR is well formed
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f'The model is invalid: {e}')
    else:
        print('The model is valid!')
    print(f"Model was stored at {onnx_model_file}")


if __name__ == "__main__":
    main()
