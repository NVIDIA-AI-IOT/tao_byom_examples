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

"""Export UNet model from smp package to ONNX."""

import os
import argparse
import onnx
import torch
import segmentation_models_pytorch as smp


# Mapping model name to the names from smp package
# tu-* models are directly loaded from timm package
encoder_dict = {
    "resnet18_unet_scratch": "resnet18",
    "resnet18_unet": "resnet18",
    "vgg16_unet_scratch": "vgg16",
    "vgg16_unet": "vgg16",
    "mobilenetv3_unet": "tu-mobilenetv3_small_100",
    "mobilenetv3_unet_scratch": "tu-mobilenetv3_small_100",
    "mobilenetv3_unet_imagenet_scratch": "tu-mobilenetv3_small_100",
}


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
    parser.add_argument('-p',
                        '--pretrained-path',
                        type=str,
                        required=False,
                        help='Pretrained weight path')
    parser.add_argument('--imagenet',
                        action='store_true',
                        help="Load ImageNet weights to encoder")
    parser.add_argument('-a',
                        '--activation',
                        type=str,
                        required=False,
                        choices=["softmax", "sigmoid", "None"],
                        default='sigmoid',
                        help='Type of activations to be used. Default: softmax')
    parser.add_argument('-x',
                        '--opset-version',
                        type=int,
                        required=False,
                        default=11,
                        help='Version of ONNX Opset. Default is set to 11')
    parser.add_argument('-c',
                        '--num-classes',
                        type=int,
                        required=False,
                        default=2,
                        help='Number of target classes')
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
    pretrained = 'scratch' in model_name
    if pretrained and (args.pretrained_path is None or not os.path.exists(args.pretrained_path)):
        raise FileNotFoundError("Requires pretrained model weight")

    opset = args.opset_version

    os.makedirs(args.output_path, exist_ok=True)
    onnx_model_file = os.path.join(args.output_path, f"{model_name}.onnx")

    if args.shape is None:
        dummy_input = torch.randn(1, 3, 320, 320)
    elif len(args.shape) == 1:
        dummy_input = torch.randn(1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        dummy_input = torch.randn(1, 3, args.shape[0], args.shape[1])
    else:
        raise ValueError('invalid input shape')

    ENCODER = encoder_dict[args.model_name]
    ENCODER_WEIGHTS = 'imagenet' if args.imagenet else None
    print(f"ENCODER Arch: {ENCODER}, ENCODER Weight: {ENCODER_WEIGHTS}")

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=args.num_classes,
        activation=args.activation,
    )
    if pretrained:
        print(f"Loading weights from {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path))

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
