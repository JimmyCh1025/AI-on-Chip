# Python/utils/codegen.py
import tvm
import os
import numpy as np

from .fuse import COMPILER_NAME
from .note import *
from tvm import relay

PATTERN_TABLE = [
    f"{COMPILER_NAME}.qconv2d_relu_maxpool",
    f"{COMPILER_NAME}.qconv2d_relu",
    f"{COMPILER_NAME}.qlinear_relu",
    f"{COMPILER_NAME}.qlinear",
    f"{COMPILER_NAME}.flatten",
    f"{COMPILER_NAME}.quantize",
    f"{COMPILER_NAME}.dequantize",
]
BUF_PREFIX = "buf_"

func_list = []


def get_node_summary(node):
    """Get a brief summary of the node, excluding children nodes."""
    if isinstance(node, relay.Call):
        # Try to read Composite or Op Name
        op_name = node.op.name if hasattr(node.op, "name") else type(node.op).__name__
        comp = node.op.attrs.get("Composite") if hasattr(node.op, "attrs") and node.op.attrs else None
        if comp:
            return f"<Call> [Composite: {comp}]"
        return f"<Call> [Op: {op_name}]"

    elif isinstance(node, relay.Function):
        return f"<Function> [Params: {len(node.params)}]"

    elif isinstance(node, relay.Constant):
        # Show shape
        shape = tuple(node.data.shape)
        return f"<Constant> shape={shape}"

    elif isinstance(node, relay.Var):
        return f"<Var> {node.name_hint}"

    elif isinstance(node, relay.Tuple):
        return f"<Tuple> length={len(node.fields)}"

    else:
        return f"<{type(node).__name__}>"


def debug_print_node(node, indent=0, prefix="|-"):
    """
    Improved Printer:
    1. List summaries of all parameters in the same layer first (Breadth-First View).
    2. Simple nodes (Constant/Var) are displayed directly without recursion.
    3. Complex nodes (Call/Function) are expanded below.
    """
    space = " " * indent
    summary = get_node_summary(node)

    # Print the current node itself
    print(f"{space}{prefix} {summary}")

    # If it is a Call or Tuple, we need to handle its arguments/fields
    children = []
    if isinstance(node, relay.Call):
        children = node.args
    elif isinstance(node, relay.Tuple):
        children = node.fields
    elif isinstance(node, relay.Function):
        # Function usually only prints the body; params are less important and can be skipped or treated as children
        children = [node.body]

    if not children:
        return

    # Phase 1: Print the "summary" of all parameters in this layer first.
    # This allows you to see the types of Arg[0]~Arg[13] in one place.
    child_indent = indent + 2
    child_space = " " * child_indent

    for i, child in enumerate(children):
        child_summary = get_node_summary(child)

        # Determine if it is a "simple node" (no need to expand)
        is_simple = isinstance(child, (relay.Constant, relay.Var))

        marker = " " if is_simple else "+"  # Mark nodes that will be expanded later with a + sign

        print(f"{child_space}Arg[{i}]: {child_summary} {marker}")

    # Phase 2: Recursively expand only "complex nodes"
    for i, child in enumerate(children):
        is_simple = isinstance(child, (relay.Constant, relay.Var))

        if not is_simple:
            print(f"{child_space} [Expanding Arg[{i}]...]")
            # Recursive call
            debug_print_node(child, indent=child_indent + 4, prefix="|-")


##############################################################################################
# Codegen
from tvm._ffi import registry
from abc import ABC, abstractmethod


class Output(dict):
    def __init__(self, name="", dtype="", need_copy=False, size=0):
        self.name = name  # Name of the variable in generated code
        self.dtype = dtype  # Data type (e.g., 'float', 'int')
        self.need_copy = need_copy  # Whether copying is required for this output
        self.size = size  # Size of the buffer or output

    def __getitem__(self, name):
        if name == "name":
            return self.name
        elif name == "dtype":
            return self.dtype
        elif name == "need_copy":
            return self.need_copy
        elif name == "size":
            return self.size
        else:
            return None

    def __repr__(self):
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "need_copy": self.need_copy,
            "size": self.size,
        }
        return str(d)


class Data(dict):
    def __init__(self):
        self.name = None  # Name of the variable in generated code
        self.dtype = None  # Data type (e.g., 'float', 'int')
        self.struct_info = None  # Size of the buffer or output
        self.data = None  # Size of the buffer or output

    def __getitem__(self, name):
        if name == "name":
            return self.name
        elif name == "dtype":
            return self.dtype
        elif name == "struct_info":
            return self.struct_info
        elif name == "data":
            return self.data
        else:
            return None


##############################################################################################
### Abstract Class
###


class CodegenCBase(ABC):
    def __init__(self):
        super().__init__()
        self.code_stream = []
        self.indent = 0

    def _append_code(self, line=""):
        """
        Helper method to append code to the code stream with proper indentation.
        """
        self.code_stream.append(" " * self.indent + line)

    def get_code(self):
        """
        Get the generated code as a single string.
        """
        return "\n".join(self.code_stream)

    def print_indents(self):
        """
        Print indents using spaces.
        """
        self._append_code()

    def enter_scope(self):
        """
        Enter a new scope, increasing the indentation.
        """
        self.indent += 2

    def exit_scope(self):
        """
        Exit the current scope, decreasing the indentation.
        """
        if self.indent < 2:
            raise ValueError("Wrong indent level detected.")
        self.indent -= 2

    def generate_backend_c_func(self, func_name, args, const_arr_name, outs):
        """
        Generate C code for the external function.
        Modified for AOT Unpacked API (Raw Pointers).
        """
        # Print signature
        self._append_code(f"int32_t {func_name}(")
        for i, arg in enumerate(args):
            dtype_str = self.get_dtype_string(arg)
            self._append_code(f"{dtype_str}* arg{i},")
        for i, out in enumerate(outs[:-1]):
            dtype_str = self.get_dtype_string(out)
            self._append_code(f"{dtype_str}* out{i},")
        self._append_code(f"{outs[-1]['dtype']}* out{len(outs) - 1})")
        self._append_code(f"{{")
        self.enter_scope()

        # Generate internal call
        self._append_code(f"{func_name}_(")
        for i, arg in enumerate(args):
            self._append_code(f"arg{i},")
        for i, out in enumerate(outs[:-1]):
            self._append_code(f"out{i},")
        self._append_code(f"out{len(outs) - 1});")
        self._append_code("return 0;")
        self.exit_scope()
        self._append_code("}")

    @abstractmethod
    def jit(self, outs):
        """
        Emit the code for external runtime.
        """
        pass

    def jit_impl(self, ext_func_id, args, body, const_arr_name, outs):
        """
        Generate the wrapper to invoke external kernels.
        """
        if const_arr_name:
            self._append_code(const_arr_name)
        self._append_code(f"\nvoid {ext_func_id}_(")

        for arg in args:
            dtype_str = self.get_dtype_string(arg)
            self._append_code(f"{dtype_str}* {arg.name_hint}, ")
        for i, out in enumerate(outs[:-1]):
            self._append_code(f"{out['dtype']}* out{i}, ")
        self._append_code(f"{outs[-1]['dtype']}* out{len(outs) - 1}) {{")
        self.enter_scope()

        # Function body
        self._append_code("")
        for stmt in body:
            self._append_code(stmt)

        # Copy output
        for i, out in enumerate(outs):
            if not out.need_copy:
                continue
            self._append_code(f"memcpy(out{i}, {out['name']}, 4 * {out['size']});")

        self.exit_scope()
        self._append_code("}")

        # Create the wrapper to call the external function
        self.generate_backend_c_func(ext_func_id, args, const_arr_name, outs)
        return self.get_code()

    def get_dtype_string(self, var):
        """
        Return the dtype string for a variable.
        """
        if isinstance(var, tvm.ir.tensor_type.TensorType):
            ttype = var.dtype
        else:
            ttype = var.checked_type.dtype

        if ttype == "float32":
            return "float"
        elif ttype == "int32":
            return "int32_t"
        elif ttype == "int64":
            return "int64_t"
        elif ttype == "int8":
            return "int8_t"
        elif ttype == "uint8":
            return "uint8_t"
        else:
            raise ValueError(f"Unsupported dtype {ttype}")

    def get_shape(self, var):
        """
        Return the shape for a variable.
        """
        return var.shape

    def create_const_var(self, symbol, const_id):
        """
        Generate a variable name for a constant variable.
        """
        return f"{symbol}_const_{const_id}"


class CSourceModuleCodegenBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create_c_source_module(self, ref):
        pass

    def get_ext_symbol(self, func):
        name_node = func.attrs["global_symbol"]
        if not name_node:
            raise ValueError("Fail to retrieve external symbol.")
        return str(name_node)


##############################################################################################
### Abstract Class
###

from typing import Tuple, List
from io import StringIO, BytesIO
from tvm.relay.expr import Call, Constant, Var, Tuple, TupleGetItem


class CodegenC(CodegenCBase):
    def __init__(self, ext_func_id: str):
        super().__init__()
        self.ext_func_id = ext_func_id
        self.func_idx = 0
        self.buf_idx = 0
        self.const_idx = 0
        self.ext_func_args = []
        self.ext_func_body = []
        self.const_array_name = ""
        self.func_decl = []
        self.const_vars = []

    def create_data_reference(self, symbol, const_id, cn):
        """
        Create a data reference for the constant.
        """
        dtype = self.get_dtype_string(cn.checked_type)
        var_name = f"{symbol}_const_{const_id}"

        var_data = Data()
        var_data.name = var_name
        var_data.dtype = dtype
        var_data.data = cn.data
        var_data.struct_info = cn.data.shape

        self.const_vars.append(var_data)
        return f"({dtype}*){var_name}"

    def get_size(self, parameter):
        if parameter == None:
            return None
        shape = self.get_shape(parameter.checked_type)
        size = 1
        for dim in shape:
            size *= dim
        return size

    def visit_expr_default(self, op):
        raise RuntimeError(f"C codegen doesn't support: {op.type_key}")

    def visit_expr(self, node):
        if isinstance(node, Var):
            return self.visit_var(node)
        elif isinstance(node, Tuple):
            return self.visit_tuple(node)
        elif isinstance(node, TupleGetItem):
            return self.visit_tuple_get_item(node)
        elif isinstance(node, Constant):
            return self.visit_constant(node)
        elif isinstance(node, Call):
            return self.visit_call(node)
        else:
            return self.visit_expr_default(node)

    def visit_var(self, node):
        self.ext_func_args.append(node)
        output = Output(name=node.name_hint, size=self.get_size(node))
        return [output]

    def visit_tuple(self, node):
        outs = []
        for field in node.fields:
            res = self.visit_expr(field)
            if len(res) != 1:
                raise RuntimeError("Tuple nesting is not supported")
            outs.append(res[0])
        return outs

    def visit_tuple_get_item(self, op):
        res = self.visit_expr(op.tuple)
        if len(res) <= op.index:
            raise RuntimeError("Index out of bounds in tuple access")
        return [res[op.index]]

    def visit_constant(self, cn):
        output = Output()
        output.name = self.create_data_reference(self.ext_func_id, self.const_idx, cn)
        dtype = self.get_dtype_string(cn.checked_type)

        if not self.const_array_name:
            self.const_array_name = ""

        if dtype not in {"float", "int32_t", "int64_t", "int8_t", "uint8_t"}:
            raise RuntimeError("Only float and int are supported for constants")

        output.dtype = dtype
        output.size = self.get_size(cn)
        const_var_name = self.create_const_var(self.ext_func_id, self.const_idx)
        self.const_idx += 1

        return [output]

    ### Get conv2D Op node information from a Compesite
    def get_conv_info(self, call):
        op_list = [
            call.op.body,
        ]

        conv2d_info = {
            "m": "DEFAULT_m",
            "e": "DEFAULT_e",
            "p": "DEFAULT_p",
            "q": "DEFAULT_q",
            "r": "DEFAULT_r",
            "t": "DEFAULT_t",
            "U": 1,
        }

        # BFS
        while len(op_list) > 0:
            op = op_list.pop(0)
            ########################################################################
            #           TODO: Extract conv2d attributes from the op node           #
            # -------------------------------------------------------------------- #
            # If the op is nn.conv2d:                                              #
            #   - Extract and store:                                               #
            #       - padding -> conv2d_info["PAD"]                                #
            #       - channels -> conv2d_info["M"]                                 #
            #       - kernel size (0 and 1) -> conv2d_info["R"], conv2d_info["S"]  #
            #   - Also set conv2d_info["m"] = conv2d_info["M"] as default.         #
            #                                                                      #
            # Hint:                                                                #
            #   - Use op.op.name to check for "nn.conv2d"                          #
            #   - Use op.attrs["padding"], op.attrs["channels"], etc.              #
            #   - You can assume padding and kernel_size are lists/tuples.         #
            #                                                                      #
            # Example:                                                             #
            #   conv2d_info["PAD"] = op.attrs["padding"][0]                        #
            ########################################################################

            # >>> Implement your code here <<<
            # Clean the following line when you finish the implementation

            if op.op.name == "nn.conv2d":
                conv2d_info["PAD"] = op.attrs["padding"][0]
                conv2d_info["M"] = op.attrs["channels"]
                conv2d_info["R"] = op.attrs["kernel_size"][0]
                conv2d_info["S"] = op.attrs["kernel_size"][1]
                conv2d_info["m"] = conv2d_info["M"]



            #raise NotImplementedError("You need to extract attributes for nn.conv2d node.")
            
            for arg in getattr(op, "args", []):
                if isinstance(arg, Call):
                    op_list.append(arg)
        return conv2d_info

    ### Traverse all the calls
    def visit_call(self, call):
        composite_name = call.op.attrs["Composite"]
        func_name = composite_name.replace(".", "_")
        in_shape = self.get_shape(call.args[0].checked_type)

        if composite_name in PATTERN_TABLE:
            print("[composite trace]", composite_name, in_shape)
        else:
            raise RuntimeError("Unrecognized composite")

        ########################################################################
        #                       TODO 1: Trace parameters                       #
        # -------------------------------------------------------------------- #
        # For each argument in call.args, determine:                           #
        #   - its mapped name in tvm_auto_args_NOTES[func_name]                #
        #   - whether it's a Constant or not                                   #
        #   - if not a constant, use `self.visit_expr(arg)` to visit it        #
        # Then fill the `parameters` dict like:                                #
        #   parameters["input"] = (value, is_const)                            #
        #                                                                      #
        # Hint:                                                                #
        #   - Use zip(call.args, tvm_auto_args_NOTES[func_name])               #
        #   - Use isinstance(arg, Constant)                                    #
        ########################################################################
        parameters = dict()

        # >>> Implement your code here <<< 
        # Clean the following line when you finish the implementation
        for arg, arg_name in zip(call.args, tvm_auto_args_NOTES[func_name]):
            is_const = isinstance(arg, Constant)

            if is_const:
                value = arg
            else:
                value = self.visit_expr(arg)[0]

            parameters[arg_name] = (value, is_const)  

        #raise NotImplementedError("Please implement argument tracing into the 'parameters' dictionary.")

        # fetch function generator
        func_gen = tvm_c_func_call_gen.get(func_name, None)
        if not func_gen:
            return parameters["input"]  # if function generator not exist, just bypass the input.

        # output buffer
        ########################################################################
        #                     TODO 2: Create output buffer                     #
        # -------------------------------------------------------------------- #
        # You need to:                                                         #
        #   - Generate a new buffer name using `BUF_PREFIX` and self.buf_idx   #
        #   - Get the output buffer size: self.get_size(call)                  #
        #   - Get the output buffer dtype:                                     #
        #     self.get_dtype_string(call.checked_type)                         #
        #                                                                      #
        # You should generate a line like:                                     #
        #   float* out_0 = (float*)malloc(size * 4);                           #
        #                                                                      #
        # Output:                                                              #
        #   - out      -> output buffer name                                   #
        #   - out_size -> total number of elements                             #
        #   - dtype    -> C-style data type                                    #
        ########################################################################

        # >>> Implement your code here <<<
        # Clean the following line when you finish the implementation

        BUF_PREFIX = "out_"
        out = f"{BUF_PREFIX}{self.buf_idx}"
        self.buf_idx += 1

        out_size = self.get_size(call)  

        dtype = self.get_dtype_string(call.checked_type)


        #raise NotImplementedError("Please implement output buffer allocation logic.")

        ### gether the parameter that we need.
        # conv2d Op info
        if "conv2d" in func_name:
            config = self.get_conv_info(call)
            config["C"] = in_shape[1]
            config["H"] = in_shape[2]
            config["W"] = in_shape[3]
        else:
            config = dict()

        # wildcard info
        for k in ["input", "weight", "bias"]:
            # default
            config[k] = None
            config[f"{k}_len"] = None
            config[f"{k}_dtype"] = None
            # get parameter
            param = parameters.get(k, None)
            if param == None:
                continue
            # unpack
            p, is_const = param
            if p == None:
                continue
            # if it is constant, now can visit it
            if is_const:
                p = self.visit_constant(p)[0]

            config[k] = p.name
            config[f"{k}_len"] = p.size
            config[f"{k}_dtype"] = p.dtype

        config["output"] = out
        config["output_len"] = out_size

        # convert quntize scale
        for k, (v, is_const) in parameters.items():
            if "scale" in k and is_const:
                n = v.data.numpy()
                config[k] = n[0] if n.ndim == 1 else n

        # malloc output buffer
        buf_create = f"{dtype}* {out} = ({dtype}*)malloc({out_size * 4});"
        self.ext_func_body.append(buf_create)

        # generate c function
        self.ext_func_body.append("".join(func_gen(config)))

        # free input buffer
        p, _ = parameters["input"]
        if BUF_PREFIX in p.name:
            buf_free = f"free({p.name});"
            self.ext_func_body.append(buf_free)

        output = Output(name=out, dtype=dtype, need_copy=True, size=out_size)
        return [output]

    def jit(self, out):
        code_stream = StringIO()
        for decl in self.func_decl:
            code_stream.write(f"{decl}\n")
        return self.jit_impl(
            self.ext_func_id,
            self.ext_func_args,
            self.ext_func_body,
            self.const_array_name,
            out,
        )


class CSourceCodegen(CSourceModuleCodegenBase):
    def __init__(self):
        super().__init__()
        self.code_stream = StringIO()
        # self.weight_c_stream = StringIO()
        # self.weight_h_stream = StringIO()
        # self.weight_bin_stream = BytesIO()

    def gen_c_func(self, func):
        """
        Generate the C function for the given Relay function.
        """
        global func_list

        if func is None:
            raise ValueError("Input error: expect a Relay function.")

        # Record the external symbol for runtime lookup
        sid = self.get_ext_symbol(func)
        func_list.append(sid)

        # Initialize the code generator
        builder = CodegenC(sid)
        out = builder.visit_expr(func.body)
        # self.code_stream.write(builder.jit(out))
        jit_code = builder.jit(out)

        return sid, builder.const_vars, jit_code

    def embed_weights(self, const_vars):
        """
        Weights are converted into C Arrays to be written to code_stream.
        Now supports conditional padding for DLA via CPU_ONLY macro.
        """
        self.code_stream.write("// --- Embedded Weights ---\n")

        for const_var in const_vars:
            data_numpy = const_var.data.numpy()
            dtype_str = const_var.dtype

            # 1. Prepare raw data (Compact Version) - For CPU usage
            flat_data_compact = data_numpy.flatten()

            # 2. Determine if padding is required (For DLA usage)
            needs_padding = False
            flat_data_padded = None

            if data_numpy.ndim == 4:
                N, C, H, W = data_numpy.shape
                remainder = C % 4
                if remainder != 0:
                    needs_padding = True
                    # print(f"[Channel Fix] Detecting padding need for {const_var.name}")
                    pad_size = 4 - remainder
                    # Create a new array padded with zeros
                    padded_data = np.zeros((N, C + pad_size, H, W), dtype=data_numpy.dtype)
                    padded_data[:, :C, :, :] = data_numpy  # Fill with original values
                    flat_data_padded = padded_data.flatten()

            # 3. Define helper function to convert numeric values to strings
            def array_to_string(arr):
                if "float" in dtype_str:
                    return ",".join([f"{x:.8f}" for x in arr])
                else:
                    return ",".join([str(x) for x in arr])

            # 4. Write C code
            # Note: Added 'aligned' attribute to ensure memory alignment safety for DLA DMA access
            self.code_stream.write(f"static const {dtype_str} {const_var.name}[] __attribute__((aligned(128))) = {{\n")

            if needs_padding:
                # --- If padding is needed, generate two data sets ---

                # Scenario A: CPU_ONLY defined, use compact data
                self.code_stream.write("#ifdef CPU_ONLY\n")
                self.code_stream.write(array_to_string(flat_data_compact))

                # Scenario B: CPU_ONLY NOT defined (DLA Mode), use padded data
                self.code_stream.write("\n#else\n")
                self.code_stream.write(array_to_string(flat_data_padded))
                self.code_stream.write("\n#endif\n")

            else:
                # --- General Case: No padding required (or not a 4D Tensor) ---
                # Both scenarios share the same data
                self.code_stream.write(array_to_string(flat_data_compact))

            self.code_stream.write("\n};\n\n")

    def create_c_source_module(self, ref):
        """
        Create a runtime module for the external library.
        """
        # Create headers
        self.code_stream.write("#include <string.h>\n")
        self.code_stream.write("#include <stdio.h>\n")
        self.code_stream.write("#include <stdlib.h>\n")
        self.code_stream.write('#include "runtime.h"\n')
        self.code_stream.write("#include <tvm/runtime/c_runtime_api.h>\n")
        self.code_stream.write("#include <dlpack/dlpack.h>\n")

        # Check that the reference is a FunctionNode
        if not isinstance(ref, tvm.relay.function.Function):
            raise TypeError("Expected a FunctionNode.")

        # Generate the C function
        sid, variables, func_code = self.gen_c_func(ref)

        self.embed_weights(variables)
        self.code_stream.write("// --- Function Logic ---\n")
        self.code_stream.write(func_code)
        full_code = self.code_stream.getvalue()

        # Debug:
        # with open("debug_full_code.c", "w") as f:
        #     f.write(full_code)

        # 6. Create CSourceModule
        # TVM Runtime will attempt to call compiler (cc/clang) to compile these code
        pf = registry.get_global_func("runtime.CSourceModuleCreate")
        if pf is None:
            raise RuntimeError("Cannot find csource module to create the external runtime module.")

        return pf(full_code, "c", [sid], [])


#############################################################################################################
# only c-codegen
@registry.register_func(f"relay.ext.{COMPILER_NAME}")
def DLA_compiler(ref):
    # print("\n[DEBUG] Inspecting IR Structure:")
    # debug_print_node(ref)
    # print("[DEBUG] Inspection Done.\n")
    assert isinstance(ref, tvm.relay.function.Function), "Expected IRModule for compilation."

    DLA_codegen = CSourceCodegen()
    result = DLA_codegen.create_c_source_module(ref)

    print("Code gen Done")

    return result
