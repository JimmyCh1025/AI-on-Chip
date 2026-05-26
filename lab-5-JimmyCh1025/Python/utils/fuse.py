# Python/utils/fuse.py
from tvm.relay import dataflow_pattern as dfp
from tvm.relay import transform
from tvm.relay import qnn
from tvm import relay
import tvm

COMPILER_NAME = "DLA"


# Define the pattern for fusion
def fuse_conv2d_bias_add_relu():
    # Define the pattern for the operations to be fused
    i = dfp.wildcard()  # Input
    w = dfp.wildcard()  # Weight
    b = dfp.wildcard()  # Bias
    dequantized_i = dfp.is_op("qnn.dequantize")(i, dfp.wildcard(), dfp.wildcard())
    dequantized_w = dfp.is_op("qnn.dequantize")(w, dfp.wildcard(), dfp.wildcard())
    dequantized_b = dfp.is_op("qnn.dequantize")(b, dfp.wildcard(), dfp.wildcard())
    conv2d_op = dfp.is_op("nn.conv2d")(dequantized_i, dequantized_w)
    bias_add_op = dfp.is_op("nn.bias_add")(conv2d_op, dequantized_b)
    relu_op = dfp.is_op("nn.relu")(bias_add_op)
    quantize_op = dfp.is_op("qnn.quantize")(relu_op, dfp.wildcard(), dfp.wildcard())
    cast_op = dfp.is_op("cast")(quantize_op)  # Assuming requantize is a cast operation
    return cast_op


def fuse_conv2d_bias_add_relu_max_pool2d():
    ######################################################################
    #         Define the pattern for the operations to be fused          #
    #            Hint: build on fuse_conv2d_bias_add_relu(),             #
    #            then dequant -> max_pool2d -> quant -> cast             #
    ######################################################################

    i = dfp.wildcard()  # Input
    w = dfp.wildcard()  # Weight
    b = dfp.wildcard()  # Bias
    
    dequantized_i = dfp.is_op("qnn.dequantize")(i, dfp.wildcard(), dfp.wildcard())
    dequantized_w = dfp.is_op("qnn.dequantize")(w, dfp.wildcard(), dfp.wildcard())
    dequantized_b = dfp.is_op("qnn.dequantize")(b, dfp.wildcard(), dfp.wildcard())
    
    conv2d_op = dfp.is_op("nn.conv2d")(dequantized_i, dequantized_w)
    bias_add_op = dfp.is_op("nn.bias_add")(conv2d_op, dequantized_b)
    relu_op = dfp.is_op("nn.relu")(bias_add_op)
    quantize_op = dfp.is_op("qnn.quantize")(relu_op, dfp.wildcard(), dfp.wildcard())
    cast_op = dfp.is_op("cast")(quantize_op)  # Assuming requantize is a cast operation
    
    dequant_op = dfp.is_op("qnn.dequantize")(cast_op, dfp.wildcard(), dfp.wildcard())
    max_pool2d_op = dfp.is_op("nn.max_pool2d")(dequant_op)
    quantize_op = dfp.is_op("qnn.quantize")(max_pool2d_op, dfp.wildcard(), dfp.wildcard())

    # cast
    cast_op2 = dfp.is_op("cast")(quantize_op)

    return cast_op2


def fuse_dense_add_relu():
    ######################################################################
    #         Define the pattern for the operations to be fused          #
    #      Hint: match dequant(i), dequant(w), dequant(b) -> dense       #
    #                  -> add -> relu -> quant -> cast                   #
    ######################################################################

    i = dfp.wildcard()  # Input
    w = dfp.wildcard()  # Weight
    b = dfp.wildcard()  # Bias
    dequantized_i = dfp.is_op("qnn.dequantize")(i, dfp.wildcard(), dfp.wildcard())
    dequantized_w = dfp.is_op("qnn.dequantize")(w, dfp.wildcard(), dfp.wildcard())
    dequantized_b = dfp.is_op("qnn.dequantize")(b, dfp.wildcard(), dfp.wildcard())

    dense_op = dfp.is_op("nn.dense")(dequantized_i, dequantized_w)
    add_op = dfp.is_op("add")(dense_op, dequantized_b)

    relu_op = dfp.is_op("nn.relu")(add_op)
    quantize_op = dfp.is_op("qnn.quantize")(relu_op, dfp.wildcard(), dfp.wildcard())
    cast_op = dfp.is_op("cast")(quantize_op)  # Assuming requantize is a cast operation
    return cast_op


def fuse_dense_add():
    ######################################################################
    #         Define the pattern for the operations to be fused          #
    #      Hint: match dequant(i), dequant(w), dequant(b) -> dense       #
    #                      -> add -> quant -> cast                       #
    ######################################################################

    i = dfp.wildcard()  # Input
    w = dfp.wildcard()  # Weight
    b = dfp.wildcard()  # Bias
    dequantized_i = dfp.is_op("qnn.dequantize")(i, dfp.wildcard(), dfp.wildcard())
    dequantized_w = dfp.is_op("qnn.dequantize")(w, dfp.wildcard(), dfp.wildcard())
    dequantized_b = dfp.is_op("qnn.dequantize")(b, dfp.wildcard(), dfp.wildcard())

    dense_op = dfp.is_op("nn.dense")(dequantized_i, dequantized_w)
    add_op = dfp.is_op("add")(dense_op, dequantized_b)

    quantize_op = dfp.is_op("qnn.quantize")(add_op, dfp.wildcard(), dfp.wildcard())
    cast_op = dfp.is_op("cast")(quantize_op)  # Assuming requantize is a cast operation
    return cast_op


def fuse_flatten():
    # Define the pattern for the operations to be fused
    dequantize_op = dfp.is_op("qnn.dequantize")(dfp.wildcard(), dfp.wildcard(), dfp.wildcard())
    flatten_op = dfp.is_op("nn.batch_flatten")(dequantize_op)
    quantize_op = dfp.is_op("qnn.quantize")(flatten_op, dfp.wildcard(), dfp.wildcard())
    cast_op = dfp.is_op("cast")(quantize_op)  # Assuming requantize is a cast operation
    return cast_op


def quantize():
    quantize_op = dfp.is_op("qnn.quantize")(dfp.wildcard(), dfp.wildcard(), dfp.wildcard())
    cast_op = dfp.is_op("cast")(quantize_op)  # Assuming requantize is a cast operation
    # return quantize_op
    return cast_op


def dequantize():
    dequantize_op = dfp.is_op("qnn.dequantize")(dfp.wildcard(), dfp.wildcard(), dfp.wildcard())
    return dequantize_op


def pattern_table():
    acc_pattern_tables = [
        (f"{COMPILER_NAME}.qconv2d_relu_maxpool", fuse_conv2d_bias_add_relu_max_pool2d()),
        (f"{COMPILER_NAME}.qconv2d_relu", fuse_conv2d_bias_add_relu()),
        (f"{COMPILER_NAME}.qlinear_relu", fuse_dense_add_relu()),
        (f"{COMPILER_NAME}.qlinear", fuse_dense_add()),
        (f"{COMPILER_NAME}.flatten", fuse_flatten()),
        (f"{COMPILER_NAME}.quantize", quantize()),
        (f"{COMPILER_NAME}.dequantize", dequantize()),
    ]
    return acc_pattern_tables


# Define the fusion function
def merge_composite_pass(mod):
    with tvm.transform.PassContext(opt_level=3):
        model_progress = dict()
        model_progress["origin"] = mod
        model_progress["MergeComposite"] = transform.MergeComposite(pattern_table())(model_progress["origin"])
        model_progress["AnnotateTarget"] = transform.AnnotateTarget([COMPILER_NAME])(model_progress["MergeComposite"])
        model_progress["MergeCompilerRegions"] = transform.MergeCompilerRegions()(model_progress["AnnotateTarget"])
        model_progress["PartitionGraph"] = transform.PartitionGraph()(model_progress["MergeCompilerRegions"])
    return (model_progress, model_progress["PartitionGraph"])
