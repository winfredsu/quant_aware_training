# Symmetric, Power-of-2 Quant-Aware-Training
We made two modifications to the original library. 

1. In `quant_ops.py`, add max_po2 and min_po2 variables in both `LastValueQuantize` and `MovingAvgQuantize`. 
2. In `quantize.py`, add 16 bit quantization for bn_folded biases. 