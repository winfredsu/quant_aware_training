# Quantization-Aware Training
This repo contains a `quantize` package modified from `tf.contrib.quantize`, and the quantization-aware training examples. It is part of the toolchain for the embedded DL engine. 

## Project Structure
- `quantize` is a modified version of `tf.contrib.quantize` package. The quantization scales are forced to be power-of-2 for pure INT8 operations on hardware. Also, we add 16-bit quantization of bias tensors. Details of implementation are described in `quantize/README.md`.
- `apps` contains the quantization-aware training examples. Including face attribute (multi-label classification), cifar10 (multi-class classification), and pet-face detection (object detection). Detailed training steps can be found in `apps/README.md`. 

## Why quant-aware training with PO2-SCALE and Symmetric Range?

### The quant-aware training method in tf.contrib.quantize
The quant-aware training method in CVPR 2018 (Google Inc.) has float scale factor and non-zero offset. With this quant strategy, real number matrix multiplication can be computed with int8 matrix multiplication and a few other operations for scaling back and zero point alignment. To run inference efficiently on CPU, the convolution should be first implemented with GEMMLP (also by google). In GEMMLP, the images are first converted with IM2COL. For example, the DWCONV in mobilenet uses 112x112 input image and 3x3 conv kernel. IM2COL first converts the input image to a 9x(112x112) matrix, and multiply it by a 1x9 kernel matrix. This method has two disadvantages: 1) the conversion adds to power and time; 2) for every 9x1 vector in the converted image matrix, the avg should be computed. This is not efficient enough for both CPU and ASIC implementations.

### Benefits of PO2-SCALE and Symmetric Range quantizations
Replace the floating-point operations involved in scaling/zero point alignment with bit shifting, so that we can use pure int hardware. 
