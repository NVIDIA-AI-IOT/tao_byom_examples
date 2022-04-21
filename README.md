# TAO BYOM Example

- [Quick Start Instructions](#quick-start-instructions)
  - [Pre-requisites](#pre-requisites)
  - [Install python dependencies](#install-python-dependencies)
- [Convert ONNX model to TAO compatible model](#convert-onnx-model-to-tao-compatible-model)
  - [Classification](classification/README.md)
    - [ResNet](classification/README.md#resnet)
    - [EfficientNet](classification/README.md#efficientnet)
  - [Semantic Segmentation](semantic_segmentation/README.md)
    - [Vanilla UNet](semantic_segmentation/README.md#vanilla-unet)
    - [ResNet UNet](semantic_segmentation/README.md#resnet-unet)
  - [Bring Your Own Layer](classification/README.md#convnext)
    - [ConvNeXt](classification/README.md#convnext)
    - [Mobilenetv3 UNet](semantic_segmentation/README.md#mobilenetv3-unet)
- [Run BYOM model through TAO Toolkit](#run-byom-model-through-tao-toolkit)
  - [Classification](notebook/classification/byom_classification.ipynb)
  - [Semantic Segmentation](notebook/semantic_segmentation/byom_unet_camvid.ipynb)
- [List of Tested Models](#list-of-tested-models)

## Quick Start Instructions
To run the reference TAO Toolkit BYOM converter implementations, follow the steps below:

### Prerequisites

Before running the examples defined in this repository, install the following items:

| **Component**  | **Version** |
| :---  | :------ |
| python |  >=3.6.9 <3.7   |
| python3-pip | >19.03.5 |
| nvidia-driver | >455 |
| nvidia-pyindex| |


### Install Python Dependencies
1. Set up the miniconda using the following instructions:

    You may follow the instructions in this [link](https://docs.conda.io/en/latest/miniconda.html) to set up a Python conda environment using miniconda.

   Once you have followed the instructions to install miniconda, set the Python version
   in the new conda environment with this command:

    ```sh
    conda create -n byom_dev python=3.6
    ```

   Once you have created this conda environemnt, you may reinstantiate it on any terminal session with this command:

   ```sh
   conda activate byom_dev
   ```

2. Install python-pip dependencies.

   These repositories relies on several third-party Python dependancies, which you can install to your conda using
   the following command.

   ```sh
   pip3 install -r requirements.txt --no-deps
   ```

3. Install TensorFlow.

   Before using the NVIDIA TAO BYOM converter, you must install TensorFlow 1.15.x. Use the following commands to install it.

   ```sh
   pip3 install nvidia-pyindex
   pip3 install nvidia-tensorflow
   ```

4. Install the NVIDIA TAO BYOM converter

   The NVIDIA TAO BYOM converter is hosted in the official PyPI repository and can be installed using the following command.

   ```sh
   pip3 install nvidia-tao-byom
   ```

5. Check your installation using the following command.

   ```sh
   tao_byom --help
   ```

## Convert ONNX Model to TAO Compatible Model
In this repository, there are currently two main taks: classification and semantic segmentation. Other taks supported by TAO Toolkit, such as Object Detection,
will be included in TAO BYOM in the future.

All the examples shown in this repository are from PyTorch. Any other deep learning frameworks that can be exported to ONNX can work as long as the data format
is channel_first `(N, C, H, W)`. If you do not wish to go through the export-to-ONNX steps, you can also start with models provided from the [ONNX models repo](https://github.com/onnx/models/tree/main/vision).
Below is a list of considerations before using TAO BYOM.

1. The ONNX model must use the `channel_first` data format.
2. Only classification and semantic segmentation are supported.
3. Dynamic input shape is not supported. You must export the ONNX model using the same input shape you will use in TAO Toolkit spec file.

## Run BYOM Model through TAO Toolkit
The end-to-end pipeline for running BYOM EfficientNet-B3 on the Pascal VOC dataset is shown in [this notebok](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/cv_samples/files), and ResNet18-UNet on the DAGM dataset is shown
in [this notebok](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/cv_samples/files).


## List of Tested Models
<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Model</th>
            <th>Source</th>
            <th>Framework</th>
            <th>Dataset</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=26>Classification</td>
            <td rowspan=3>ResNet</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py>timm </a></td>
            <td> PyTorch</td>
            <td rowspan=24> <a href="https://www.image-net.org/download.php">ImageNet1K</a></td>
        </tr>
        <tr>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision/classification/resnet>ONNX/models </a></td>
            <td> ONNX (MXNet)</td>
        </tr>
                <tr>
            <td rowspan=2>EfficientNet</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py>timm </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/lukemelas/EfficientNet-PyTorch>EfficientNet-PyTorch </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td rowspan=2>VGG</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision>ONNX/models </a></td>
            <td> ONNX (MXNet)</td>
        </tr>
        <tr>
            <td rowspan=3>MobileNetv2</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py>timm </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision/classification/mobilenet>ONNX/models </a></td>
            <td> ONNX (MXNet)</td>
        </tr>
        <tr>
            <td rowspan=2>SqueezeNet</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision/classification/squeezenet>ONNX/models </a></td>
            <td> ONNX (Caffe2)</td>
        </tr>
        <tr>
            <td rowspan=2>ShuffleNet</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision/classification/shufflenet>ONNX/models </a></td>
            <td> ONNX (Caffe2)</td>
        </tr>
        <tr>
            <td rowspan=1>CSPDarkNet</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cspnet.py>timm </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td rowspan=2>DenseNet</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision/classification/densenet-121>ONNX/models </a></td>
            <td> ONNX (Caffe2)</td>
        </tr>
        <tr>
            <td rowspan=2>GoogleNet</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td><a href=https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet>ONNX/models </a></td>
            <td> ONNX (Caffe2)</td>
        </tr>
        <tr>
            <td>Inceptionv3</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td rowspan=1>EfficientNetv2</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>timm </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td rowspan=1>RegNet</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td rowspan=1>ConvNeXt</td>
            <td><a href=https://pytorch.org/vision/stable/models.html>torchivsion </a></td>
            <td> PyTorch</td>
        </tr>
        </tr>
        <tr>
            <td rowspan=1>MobileNetv3</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py>timm </a></td>
            <td> PyTorch</td>
        </tr>
        <tr>
            <td rowspan=1>MobileNetv3</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models>timm </a></td>
            <td> PyTorch</td>
            <td rowspan=1> <a href="https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md"> ImageNet21K </a> </td>
        </tr>
        <tr>
            <td rowspan=1>ResNext</td>
            <td><a href=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py>timm </a></td>
            <td> PyTorch</td>
            <td rowspan=1> <a href="https://arxiv.org/abs/1905.00546"> IG-3.5B </a></td>
        </tr>
        <tr>
            <td rowspan=3>Semantic Segmentation</td>
            <td>Vanilla UNet</td>
            <td><a href="https://github.com/milesial/Pytorch-UNet">PyTorch-UNet</a></td>
            <td> PyTorch</td>
            <td> <a href="https://www.kaggle.com/c/carvana-image-masking-challenge">Carvana</a></td>
        </tr>
        <tr>
            <td>VGG16-UNet</td>
            <td><a href="https://github.com/qubvel/segmentation_models.pytorch">segmentation_models.pytorch</a></td>
            <td> PyTorch</td>
            <td rowspan=2> <a href="http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/"> CamVid</td>
        </tr>
        <tr>
            <td>ResNet18-UNet</td>
            <td><a href="https://github.com/qubvel/segmentation_models.pytorch">segmentation_models.pytorch</a></td>
            <td> PyTorch</td>
        </tr>
    </tbody>
</table>
