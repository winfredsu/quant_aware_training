# Quant-Aware Training Guide

## Prerequisites
- Tensorflow version = 1.13.1
- Put the `quantize` package at a proper position, or modify the syspaths in `train.py` and `freeze.py`.

## Training Steps
### 0. Build dataset and model
Customize `config.py`, `model.py` and `dataset.py`. And use proper prediction function in `train.py` and `test.py` since the model constructed in `model.py` outputs only logits. 

### 1. Train with float32 until we get a good accuracy
Train with float model. If the loss doesn't decrease after some steps, try a smaller lr_start.

`./train.py --train_dir=./train --train_step_max=20000 --val_step_interval=1000 --save_step_interval=1000 --lr_start=0.045`

### 2. Finetune with quantization enabled
After getting a satisfying accuracy, start finetuning with quantization-aware training enabled. Run the following command for example. 

`./train.py --train_dir=./train --quantize=True --start_ckpt=model.ckpt-400 --is_first_finetuning=True --train_step_max=8000 --val_step_interval=500 --save_step_interval=500`

Note that the `--is_first_finetuning` will reset the global_step to get proper finetuning learning_rate. So set this parameter `False` if we are training from a ckpt that has already been quantized.

### 3. Freeze
Create an inference graph and freeze the parameters. This step also performs batchnorm folding.

`./freeze.py --train_dir=./train --output_dir=. --ckpt=model.quant.ckpt-20000 --output_node=MobilenetV1/Squeeze --frozen_pb_name=mobilenet_v1_128_0.25_cifar10.pb`

### 4. Test the frozen model
`./test.py --frozen_pb=mobilenet_v1_128_0.25_cifar10.pb --output_node=MobilenetV1/Squeeze`
