# Quantization-Aware Training
This repo contains a `quantize` package modified from `tf.contrib.quantize`, and the quantization-aware training examples. It is part of the toolchain for the embedded DL engine. 

## Walkthrough
- `quantize` is a modified version of `tf.contrib.quantize` package. The quantization scales are forced to be power-of-2 for pure INT8 operations on hardware. Also, we add quantization of bias tensors. Details of implementation are described in `quantize/README.md`.
- `apps` contains the quantization-aware training examples. Including face attribute (multi-label classification) and cifar10 (multi-class classification).  

## Why quant-aware training with PO2-SCALE and Symmetric Range?

### The quant-aware training of tf.contrib.quantize
The quant-aware training method in CVPR 2018 (Google Inc.) [] has float scale factor and non-zero offset. With this quant strategy, real number matrix multiplication can be computed with int8 matrix multiplication and a few other operations for scaling back/zero point. To run inference efficiently on CPU, we should first implement Conv with GEMMLP (also by google). In GEMMLP, the images are first converted with IM2COL. For example, the DWCONV in mobilenet uses 112x112 input image and 3x3 conv kernel. IM2COL first converts the input image to a 9x(112x112) matrix, and multiply it by a 1x9 kernel matrix. This method has two disadvantages: 1) the conversion adds to power and time; 2) for every 9x1 vector in the converted image matrix, the avg should be computed. 1
