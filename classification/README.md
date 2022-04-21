# Classification Examples

## ResNet
[ResNet](https://arxiv.org/abs/1512.03385) is one of the most popular networks for various computer vision tasks.

### Export PyTorch Model to ONNX
You will use ResNet from Torchvision that has been trained on ImageNet. The model can be easily exported to ONNX using the below command.

```sh
python export_torchvision.py -m resnet18
```

Once the export is complete, an ONNX file named `resnet18.onnx` will be created. You can use `netron` to see how the ONNX model looks.

```sh
netron onnx_models/resnet18.onnx
```

![resnet18](/assets/resnet18.png)

If you wish to fine tune the pretrained model on a different dataset through TAO Toolkit, you must remove the classification head for ImageNet.
Hence, the final converted TAO model should only contain layers up to the penultimate layer, which is a layer before the average pooling. In this case, the node name is `188`.

### Run Conversion
Now you are ready to run conversion. Here, the `-p` option specifies the ONNX node that corresponds to the penultimate layer.

```sh
tao_byom -m onnx_models/resnet18.onnx -r results/resnet18 -n resnet18 -k nvidia_tlt -p 188
```

The model should convert correctly without any errors and be saved as `results/resnet18/resnet18.tltb`.


## EfficientNet
[EfficientNet](https://arxiv.org/abs/1905.11946) is another popular computer vision network for classification. The following steps will illustrate how to
export EfficientNet to a model that is TAO compatible.

### Export PyTorch Model to ONNX
For EfficientNet, you will use the model weights and implementation from [timm](https://github.com/rwightman/pytorch-image-models). EfficientNet B3 was
trained using an input shape of (320, 320), so you must also specify the input shape:

```sh
python export_timm.py -m efficientnet_b3 --shape 320
```

Once completed, an ONNX file named `efficientnet_b3.onnx` will be created. You can use `netron` to see how the ONNX model looks.

```sh
netron onnx_models/efficientnet_b3.onnx
```

![efficientnetb3](/assets/efficientnet_b3.png)

Similar to the ResNet example, if you wish to fine tune using this model on a different dataset through TAO Toolkit, you must specify the penultimate node as `1035`.

### Run Conversion

```sh
tao_byom -m onnx_models/efficientnet_b3.onnx -r results/efficientnet_b3 -n efficientnet_b3 -k nvidia_tlt -p 1035
```

The model should convert correctly without any errors and be saved as `results/efficientnet_b3/efficientnet_b3.tltb`.

## ConvNeXt
[ConvNeXt](https://arxiv.org/abs/2201.03545), which was published in CVPR 2022, demonstrates another important feature of BYOM converter. It is a state-of-the-art
architecture with combination of standard convolutions and multi-head self-attentions (MHSA) from transformers and achieves good performance on ImageNet Classification. ConvNeXt was specifically chosen as it contains layers that are currently not supported by TAO BYOM converter.

`Erf` and `ReduceProd` are used as operations inside ConvNeXt, but they are not supported in TAO BYOM converter. Hence, they can serve as good examples to demonstrate how you
can bring your own layer into the TAO BYOM converter.

### Export PyTorch Model to ONNX
First, you need to export a pretrained model from PyTorch to ONNX so the TAO BYOM converter can generate a TAO compatible model. Model weights and implementation from
the [timm](https://github.com/rwightman/pytorch-image-models) package will be used. Run the below command to download the pretrained model and export it to ONNX.

```sh
python export_timm.py -m convnext_tiny
```

Once completed, an ONNX file named `convnext_tiny.onnx` will be created. You can use `netron` to see how the ONNX model looks.

```sh
netron onnx_models/convnext_tiny.onnx
```

Since this model adds MHSA to later stage of the network, finding the penultimate node can be tricky. If you refer to the [original PyTorch implementation](https://github.com/rwightman/pytorch-image-models/blob/f670d98cb8ec70ed6e03b4be60a18faf4dc913b5/timm/models/convnext.py#L313), you can see that the `forward_head` starts with `GlobalAveragePool`. Hence, if you wish to fine tune using this model on a different dataset through TAO Toolkit, you must specify the penultimate node as `875`.

### Bring your Own Layer
Next, convert ConvNeXt through the TAO BYOM converter.

```sh
tao_byom -m onnx_models/convnext_tiny.onnx -r results/convnext_tiny -n convnext_tiny -k nvidia_tlt -p 875
```

You will see an output like below:

```sh
INFO: Converter is called.
ERROR: These operators are not supported in our converter ['ReduceProd', 'Erf']
```

As a result, you will need to use your own implementation of `ReduceProd` and `Erf`.

### Writing a Custom Keras Layer
In `convnext/erf.py`, you can see that the custom ERFLayer has been already implemented for you. If you wish to understand more about writing a custom layer
in Keras, refer to the [official documentation](https://keras.io/guides/making_new_layers_and_models_via_subclassing/).

Keras does not support Erf natively so TF's `tf.math.erf` can be used to compute Erf. The benfit of creating a custom layer is that you can utilize TF functionalities that are not supported by the native Keras.

Once completed, update the file information in `convnext/custom_meta.json`.

### Conversion Functions to Register to TAO BYOM
The conversion function to register to the TAO BYOM converter is already present in `convnext/user_generated_layer.py`. Follow the below steps to convert an ONNX
node into a Keras layer.

1. Get all the necessary attributes and inputs of a node.
2. Convert numpy inputs from the node to TF tensors.
3. Call a Keras layer that corresponds to the node.
4. Pass the output of the Keras layer to `layers[node_name]`.
5. If a lambda function or custom Keras layer has been used, specify it in `lambda_func[node_name]`

Inside `convnext/user_generated_layer.py`, `convert_reduce_prod` function illustrates an example of using `keras.layers.Lambda` layer in conjunction with `keras.backend` operation to implement `ReduceProd` in Keras. A detailed comments are added to the code for your further understanding.

Once completed, update the file information in `convnext/custom_meta.json`.

### Run conversion
Now that you have all the implementations in place, run the conversion again.

```sh
PYTHONPATH=${PWD} tao_byom -m onnx_models/convnext_tiny.onnx -r results/convnext_tiny -n convnext_tiny -c convnext/custom_meta.json -k nvidia_tlt -p 875
```

The model should convert correctly without any errors and be saved as `results/convnext_tiny/convnext_tiny.tltb`.
