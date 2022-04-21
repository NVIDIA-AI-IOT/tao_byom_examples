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

"""Export classification model from torchvision to ONNX."""

import os
import onnx
import argparse
import torchvision
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
    parser.add_argument("-fs",
                        '--from_scratch',
                        action='store_true',
                        help="Not load pretrained_weights.")
    return parser.parse_args()


def main(args=None):
    """Run the training process."""
    args = parse_command_line(args)
    model_name = args.model_name

    print("Model Name: ", model_name)
    if "_scratch" in model_name:
        orig_model_name = model_name
        model_name = model_name.replace("_scratch", "")
    else:
        orig_model_name = model_name
    pretrained = not args.from_scratch

    opset = args.opset_version

    os.makedirs(args.output_path, exist_ok=True)
    onnx_model_file = os.path.join(args.output_path, f"{orig_model_name}.onnx")

    try:
        # from torchvision
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    except AttributeError:
        print(f"{model_name} is not part of torchvision")

    dummy_input = torch.randn(1, 3, 224, 224)

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
