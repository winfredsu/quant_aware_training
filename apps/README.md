# Quant-Aware Training Guide
For image classification tasks, the scripts in this repo is self-contained. For object-detection applications, the object_detection API of tensorflow is required.

In the provided examples, `cifar10`, `faceattr` and `indicator_diagram` are image classification examples, and `pet_face` is an object detection example using ssd-ppn detector.

## Image Classification
### Prerequisites
- Tensorflow version = 1.13.1
### Training Steps
#### 0. Build dataset and model
Customize `config.py`, `model.py` and `dataset.py` in the application directory. And use proper prediction function in `train.py` and `test.py` since the model constructed in `model.py` outputs only logits. 

#### 1. Train with float32 until we get a good accuracy
Train with float model. If the loss doesn't decrease after some steps, try a smaller lr_start.

`./train.py --train_dir=./train --train_step_max=20000 --val_step_interval=1000 --save_step_interval=1000 --lr_start=0.045`

#### 2. Finetune with quantization enabled
After getting a satisfying accuracy, start finetuning with quantization-aware training enabled. Run the following command for example. 

`./train.py --train_dir=./train --quantize=True --start_ckpt=model.ckpt-400 --is_first_finetuning=True --train_step_max=8000 --val_step_interval=500 --save_step_interval=500`

Note that the `--is_first_finetuning` will reset the global_step to get proper finetuning learning_rate. So set this parameter `False` if we are training from a ckpt that has already been quantized.

#### 3. Freeze
Create an inference graph and freeze the parameters. This step also performs batchnorm folding.

`./freeze.py --train_dir=./train --output_dir=. --ckpt=model.quant.ckpt-20000 --output_node=MobilenetV1/Squeeze --frozen_pb_name=mobilenet_v1_128_0.25_cifar10.pb`

#### 4. Test the frozen model
`./test.py --frozen_pb=mobilenet_v1_128_0.25_cifar10.pb --output_node=MobilenetV1/Squeeze`

## Object Detection
For object detection tasks, follow the instructions in [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

Before training, we should **replace** the original quantize lib with the provided one in this repo. 

### 1. Prepare the dataset
Here we use the Oxford-IIIT Pet dataset. When creating tf record, we use only two classes `cat` and `dog`. 

### 2. Config the pipline
In this example, we use mobilenet as the base net and ssd-ppn as the detector. 

### 3. Train with floating-point number
Start from a mobilenet ckpt trained on Imagenet. The graph_rewriter is recommended to be disabled during floating-point training for higher training speed.

### 4. Train with Quant-Aware-Training
Add the graph_rewriter in the pipeline configuration.

### 5. Export tflite graph
Run `sf_export_lite.sh` to export an inference graph. 
