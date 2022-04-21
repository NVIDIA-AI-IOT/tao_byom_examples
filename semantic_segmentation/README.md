# Semantic Segmentation Examples

## Vanilla UNet
[UNet](https://arxiv.org/abs/1505.04597) is an encoder-decoder architecture that is widely adapted for semantic segmentation tasks. In this example,
we will use Vanilla UNet trained on the Kaggle [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).

### Export PyTorch Model to ONNX
The model weights and implementation are adapted from [PyTorch-UNet](https://github.com/milesial/Pytorch-UNet), which has a pretrained model uploaded to `torch.hub`.
To obtain the ONNX model, simply run the below command.

```sh
python export_vanilla.py -m vanilla_unet
```

Once completed, an ONNX file named `vanilla_unet.onnx` will be created. You can use `netron` to see how the ONNX model looks.

```sh
netron onnx_models/vanilla_unet.onnx
```

![vanilla_unet](/assets/vanilla_unet.png)

For UNet, it is important that the converted TAO model only output **logits** and not include any activations at the end (e.g. sigmoid/softmax). With Vanilla UNet,
you can see that there are no activations at the end.

### Run Conversion
To convert the ONNX model to a TAO compatible model, run the below command.

```sh
tao_byom -m onnx_models/vanilla_unet.onnx -r results/vanilla_unet -n vanilla_unet -k nvidia_tlt 
```

The model should convert correctly without any errors and be saved as `results/vanilla_unet/vanilla_unet.tltb`.

## Resnet UNet
Instead of simple encoder in Vanilla UNet, you can utilize the ImageNet pretrained weights in the encoder for faster training convergence. Hence, you will use ResNet18
pretrained on ImageNet as the encoder.

### Export PyTorch Model to ONNX
For ResNet18 UNet, you will use model weights and architectures from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). The main benefit
of this package is that it allows users to specify different types of encoders supported by [timm](https://github.com/rwightman/pytorch-image-models).

The [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library does not provide any pretrained weights for the UNet decoder. If you wish to
train a UNet model, refer to [this example](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb), which illustrates
how to train a model on [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). However, this is out of scope for this repository, which does not cover
model training in PyTorch. Instead, you will export an ONNX model without any pretrained weights in the decoder. To do so, pass a model name with a pattern like `*_scratch` at
the end so that no weights are passed for the decoder.

```sh
python export_smp.py -m resnet18_unet_scratch
```

If you wish to pass pretrained weights of your own, you may pass the path of the PyTorch weights through the `-p` argument. Once completed, an ONNX file named `resnet18_unet_scratch.onnx`
will be created. You can use `netron` to see how the ONNX model looks.

```sh
netron onnx_models/resnet18_unet_scratch.onnx
```

![resnet18_unet](/assets/resnet18_unet.png)

You can see that the exported model includes `sigmoid` at the end. Hence, you must run conversion without this node for the model to be compatible with TAO Toolkit.

### Run Conversion
To convert the ONNX model to a TAO-compatible model, run the below command.

```sh
tao_byom -m onnx_models/resnet18_unet_scratch.onnx -r results/resnet18_unet_scratch -n resnet18_unet_scratch -k nvidia_tlt -p 308
```

The model should convert correctly without any errors and be saved as `results/resnet18_unet_scratch/resnet18_unet_scratch.tltb`.

## MobileNetv3 UNet
[MobileNetV3](https://arxiv.org/abs/1905.02244), which was published in ICCV 2019, is an efficient architecture with good performance on ImageNet Classification. This model will be used as the encoder to the UNet architecture. MobileNetV3 was specifically chosen as it contains a layer that is currently
not supported by TAO BYOM converter.

`HardSigmoid` is used as an activation inside MobileNetV3, but is not supported in TAO BYOM converter. Hence, it can serve as a good example to demonstrate how you
can bring your own layer into the TAO BYOM converter.

### Export PyTorch model to ONNX
Like Resnet18 UNet, the model weights and architectures of MobileNetv3 UNet are taken from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).
You will use this model in the [BYOM UNet example notebook](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/cv_samples/files) on NGC. You will export the ONNX model to train on the [DAGM dataset](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection). There are two additional arguments to export
models with the input shape and last activation layer targeted for training on the DAGM dataset.

Run the following model conversion:

```sh
python export_smp.py -m mobilenetv3_unet_scratch --shape 512 --activation softmax
```

If you wish to pass your own pretrained weights, you can use the `-p` argument to specify the path of the PyTorch weights. Once conversion is complete, an ONNX file named
`mobilenetv3_unet_scratch.onnx` will be created. You can use `netron` to see how the ONNX model looks.

```sh
netron onnx_models/mobilenetv3_unet_scratch.onnx
```

![mobilenetv3_unet](/assets/mobilenetv3_unet.png)

If you click on one of the `HardSigmoid` nodes, you will see the node properties similar to those shown below.
![hardsigmoid_prop](/assets/hardsigmoid_property.png)

By referring to the [ONNX documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid), you can see that the value of alpha has been changed from
the default value of 0.2 to 0.167. The 0.167 value is the default in [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html), while the
default ONNX value of 0.2 is based on [TensorFlow](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/activations/hard_sigmoid). Hence, you need to define
a custom Keras layer for hard sigmoid implementation.

Note that the exported model includes `softmax` at the end. Hence, you must run conversion without this node so the model compatible with TAO Toolkit.

### Bring your Own Layer
Next, convert MobileNetV3 through the TAO BYOM converter.

```sh
tao_byom -m onnx_models/mobilenetv3_unet_scratch.onnx -r results/mobilenetv3_unet_scratch -n mobilenetv3_unet_scratch -k -k nvidia_tlt -p 531
```

You will see an output like below:

```sh
INFO: Converter is called.
WARNING: These operators are not supported in our converter ['HardSigmoid']
```

As a result, you will need to use your own implementation of `HardSigmoid`.

### Writing a Custom Keras Layer
In `mobilenetv3/hard_sigmoid.py`, you can see that the custom HardSigmoidLayer has been already implemented for you. If you wish to understand more about writing a custom layer
in Keras, refer to the [official documentation](https://keras.io/guides/making_new_layers_and_models_via_subclassing/).

This example does not use the native `keras.activations.hard_sigmoid` because the API does not allow the alpha and beta value to be set manually. Hence, this example uses `tf.clip_by_value`
instead to directly compute the hard sigmoid. The benfit of creating a custom layer is that you can utilize TF functionalities that are not supported by the native Keras.

Once completed, update the file information in `mobilenetv3/custom_meta.json`.

### Conversion Functions to Register to TAO BYOM
The conversion function to register to the TAO BYOM converter is already present in `mobilenetv3/user_generated_layer.py`. Follow the below steps to convert an ONNX
node into a Keras layer.

1. Get all the necessary attributes and inputs of a node.
2. Convert numpy inputs from the node to TF tensors.
3. Call a Keras layer that corresponds to the node.
4. Pass the output of the Keras layer to `layers[node_name]`.
5. If a lambda function or custom Keras layer has been used, specify it in `lambda_func[node_name]`

Once completed, update the file information in `mobilenetv3/custom_meta.json`.

### Run Conversion
Now that you have all the implementations in place, run the conversion again.

```sh
PYTHONPATH=${PWD} tao_byom -m onnx_models/mobilenetv3_unet_scratch.onnx -r results/mobilenetv3_unet_scratch -n mobilenetv3_unet_scratch -c mobilenetv3/custom_meta.json -k nvidia_tlt -p 531
```

The model should convert correctly without any errors and be saved as `results/mobilenetv3_unet_scratch/mobilenetv3_unet_scratch.tltb`.
