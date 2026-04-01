import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from layer_info import (
    ShapeParam,
    Conv2DShapeParam,
    LinearShapeParam,
    MaxPool2DShapeParam,
)

from lib.models.vgg import VGG
from network_parser import torch2onnx


def parse_pytorch(model: nn.Module, input_shape=(1, 3, 32, 32)) -> list[ShapeParam]:
    layers = []
    #! <<<========= Implement here =========>>>

    # Define a hook to intercept tensor shapes during the forward pass
    def hook_fn(module, input, output):
        in_tensor = input[0]
        class_name = module.__class__.__name__

        # Tensor of PyTorch format is (batch, channel, height, width) 
        if "Conv" in class_name:
            layer = Conv2DShapeParam(
                N=in_tensor.shape[0],  # batch size
                H=in_tensor.shape[2],  # input height
                W=in_tensor.shape[3],  #  ''   width
                R=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,  # filter height
                S=module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size,  # filter width
                E=output.shape[2],     # output height
                F=output.shape[3],     # output width
                C=in_tensor.shape[1],  # input channels
                M=output.shape[1],     # output channels
                U=module.stride[0] if isinstance(module.stride, tuple) else module.stride, # stride
                P=module.padding[0] if isinstance(module.padding, tuple) else module.padding # padding
            )
            layers.append(layer)
        
        elif "Linear" in class_name:
            layer = LinearShapeParam(
                N=in_tensor.shape[0],
                in_features=module.in_features,
                out_features=module.out_features
            )
            layers.append(layer)

        elif "MaxPool" in class_name:
            layer = MaxPool2DShapeParam(
                N=in_tensor.shape[0],
                kernel_size=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                stride=module.stride[0] if isinstance(module.stride, tuple) else module.stride
            )
            layers.append(layer)
            

    # 1. Register the hook using register_forward_hook
    hooks = []
    for module in model.modules():
        class_name = module.__class__.__name__
        # check module type if it is Conv2d or MaxPool2d or Linear return true, else return false
        if "Conv" in class_name or "MaxPool" in class_name or "Linear" in class_name:
            hooks.append(module.register_forward_hook(hook_fn))

    # 2. Run a dummy forward pass to trigger the hooks
    # Create a input tensor
    dummy_input = torch.randn(1, 3, 32, 32)

    # 3. Perform the forward pass, triggering the hook
    model(dummy_input)

    # 4. Remove the hooks to prevent them from being triggered on every forward pass
    for hook in hooks:
        hook.remove()
    
    return layers


def parse_onnx(model: onnx.ModelProto) -> list[ShapeParam]:
    layers = []
    #! <<<========= Implement here =========>>>
    inferred_model = onnx.shape_inference.infer_shapes(model)

    '''
    all_op_types = set()
    for node in inferred_model.graph.node:
        all_op_types.add(node.op_type)
    
    print(f"ONNX model contain operator types: {all_op_types}")
    '''

    def _get_tensor_shape(tensor_name: str):

        # Search for the tensor with the given name
        for value_info in inferred_model.graph.value_info:
            if value_info.name == tensor_name:
                return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        # If not found; search the model's inputs
        for input_info in inferred_model.graph.input:
            if input_info.name == tensor_name:
                return [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]

        # If still not found; search the model's outputs
        for output_info in inferred_model.graph.output:
            if output_info.name == tensor_name:
                return [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]

        return []

    for node in inferred_model.graph.node:
        if node.op_type == "Conv":
            # extract shapes
            in_shape = _get_tensor_shape(node.input[0])
            out_shape = _get_tensor_shape(node.output[0])
            
            # extract attributes
            attrs = {attr.name: attr for attr in node.attribute}
            kernel_shape = attrs['kernel_shape'].ints
            strides = attrs['strides'].ints if 'strides' in attrs else [1, 1]
            pads = attrs['pads'].ints if 'pads' in attrs else [0, 0, 0, 0]
            
            # build data class
            layer = Conv2DShapeParam(
                N=in_shape[0] if in_shape else 1,
                H=in_shape[2] if len(in_shape)>2 else 0,
                W=in_shape[3] if len(in_shape)>3 else 0,
                R=kernel_shape[0],
                S=kernel_shape[1], 
                E=out_shape[2] if len(out_shape)>2 else 0,
                F=out_shape[3] if len(out_shape)>3 else 0,
                C=in_shape[1] if len(in_shape)>1 else 0,
                M=out_shape[1] if len(out_shape)>1 else 0,
                U=strides[0],
                P=pads[0]  # Assuming symmetric padding 
            )
            layers.append(layer)
            
        elif node.op_type == "MaxPool":
            in_shape = _get_tensor_shape(node.input[0])
            attrs = {attr.name: attr for attr in node.attribute}
            kernel_shape = attrs['kernel_shape'].ints
            strides = attrs['strides'].ints if 'strides' in attrs else [1, 1]
            
            layer = MaxPool2DShapeParam(
                N=in_shape[0] if in_shape else 1,
                kernel_size=kernel_shape[0],
                stride=strides[0]
            )
            layers.append(layer)
            
        elif node.op_type == "Gemm":
            # PyTorch's nn.Linear usually exports to the ONNX "Gemm" operator
            in_shape = _get_tensor_shape(node.input[0])
            weight_shape = _get_tensor_shape(node.input[1])
            
            attrs = {attr.name: attr for attr in node.attribute}
            transB = attrs['transB'].i if 'transB' in attrs else 0
            
            # If transB is 1 (default for PyTorch linear layer exports), weights are [out_features, in_features]
            if transB and weight_shape:
                in_f = weight_shape[1]
                out_f = weight_shape[0]
            elif weight_shape:
                in_f = weight_shape[0]
                out_f = weight_shape[1]
            else:
                in_f = in_shape[-1] if in_shape else 0
                out_f = _get_tensor_shape(node.output[0])[-1]

            layer = LinearShapeParam(
                N=in_shape[0] if in_shape else 1,
                in_features=in_f,
                out_features=out_f
            )
            layers.append(layer)
    return layers


def compare_layers(answer, layers):
    if len(answer) != len(layers):
        print(
            f"Layer count mismatch: answer has {len(answer)}, but ONNX has {len(layers)}"
        )

    min_len = min(len(answer), len(layers))

    for i in range(min_len):
        ans_layer = vars(answer[i])
        layer = vars(layers[i])

        diffs = {
            k: (ans_layer[k], layer[k])
            for k in ans_layer
            if k in layer and ans_layer[k] != layer[k]
        }

        if diffs:
            print(f"Difference in layer {i + 1} ({type(answer[i]).__name__}):")
            for k, (ans_val, val) in diffs.items():
                print(f"  {k}: answer = {ans_val}, onnx = {val}")

    if len(answer) > len(layers):
        print(f"Extra layers in answer: {answer[len(layers) :]}")
    elif len(layers) > len(answer):
        print(f"Extra layers in yours: {layers[len(answer) :]}")


def run_tests() -> None:
    """Run tests on the network parser functions."""
    answer = [
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=64, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=64, M=192, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=192, M=384, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=384, M=256, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=256, M=256, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        LinearShapeParam(N=1, in_features=4096, out_features=256),
        LinearShapeParam(N=1, in_features=256, out_features=128),
        LinearShapeParam(N=1, in_features=128, out_features=10),
    ]

    # Test with the PyTorch model.
    model = VGG()
    layers_pth = parse_pytorch(model)

    # Define the input shape.
    dummy_input = torch.randn(1, 3, 32, 32)
    # Save the model to ONNX.
    torch2onnx(model, "parser_onnx.onnx", dummy_input)
    # Load the ONNX model.
    model_onnx = onnx.load("parser_onnx.onnx")
    layers_onnx = parse_onnx(model_onnx)

    # Display results.
    print("PyTorch Network Parser:")
    if layers_pth == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_pth)

    print("ONNX Network Parser:")
    if layers_onnx == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_onnx)

'''class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 30 * 30, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''

if __name__ == "__main__":

    '''model = SimpleModel()
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, "simple_model.onnx")
    
    import onnxruntime as ort
    session = ort.InferenceSession("simple_model.onnx")
    result = session.run(None, {"x": dummy_input.numpy()})
    print("ONNX 輸出張量的形狀是:", result[0].shape)
    '''
    run_tests()
