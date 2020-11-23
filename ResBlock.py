import tensorflow as tf
import numpy as np

"""
This function performs forward pass through 2 layers and performs
residual addition
"""

def res_block(input,layer_1,layer_2):
    layer_1_out = layer_1(input)
    layer_2_out = layer_2(layer_1_out)
    res_add = layer_2_out+input
    return res_add
