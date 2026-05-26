# Python/utils/utils.py
import tvm
import onnx
from tvm import relay

from tvm.micro import export_model_library_format

from tvm.relay.backend import Runtime
from tvm.contrib import cc as _cc
import os
from tvm import runtime as tvm_runtime
from tvm.target import Target
from .codegen import func_list


def load_onnx_model(model_path, input_name="input", input_shapes=[1, 3, 32, 32]):
    input_info = {input_name: input_shapes}
    onnx_model = onnx.load_model(model_path)
    relay_mod, params = relay.frontend.from_onnx(onnx_model, input_info)
    return relay_mod, params


def dump_relay_model(relay_model, path):
    with open(path, "w") as f:
        f.write(repr(relay_model))


def build_model(relay_model, mod_params, opts):
    runtime = Runtime("crt", {"system-lib": False})
    target = "c"

    executor = relay.backend.Executor(
        "aot",
        {
            "unpacked-api": True,
            "interface-api": "packed",
            "link-params": True,
        },
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = relay.build(
            relay_model,
            target=target,
            runtime=runtime,
            params=mod_params,
            executor=executor,
        )

    print("Relay build complete.")

    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    # Export all
    # tvm.micro.export_model_library_format(lib, "output/all.tar")
    # print("Exported MLF to output/all.tar")

    # Export the Library.
    #####################################
    # export source code implementation #
    #####################################
    ## target = c
    ## Since the CSourceModule is already compiled, export_library packages everything.
    # lib_file_name = os.path.join(build_dir, "model.tar")
    # lib.get_lib().export_library(lib_file_name)

    # print(f"Library exported to {lib_file_name}")
    # ------------------------------------

    ####################################
    #   export lib.so implementation   #
    ####################################
    ## target = llvm
    workspace_dir = "output/lib_objects"
    output_lib_path_cpu = "output/lib_cpu.so"
    output_lib_path_dla = "output/lib_dla.so"

    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    root_dir = os.path.dirname(os.path.abspath(opts.out_dir))
    include_dir = os.path.join(root_dir, "simulation", "software", "include", "eyeriss")

    compile_options_cpu = [
        "-shared",
        "-fPIC",
        "-O3",
        "-DCPU_ONLY",
        "-Wno-implicit-function-declaration",
        "-I" + include_dir,
    ]

    compile_options_dla = [
        "-shared",
        "-fPIC",
        "-O3",
        "-Wno-implicit-function-declaration",
        "-I" + include_dir,
    ]

    # Export lib_cpu.so for cpu's runtime APIs
    lib.export_library(
        output_lib_path_cpu,
        workspace_dir=workspace_dir,
        options=compile_options_cpu,
        cc="gcc",
    )
    # Export lib_dla.so for dla's runtime APIs
    lib.export_library(
        output_lib_path_dla,
        workspace_dir=workspace_dir,
        options=compile_options_dla,
        cc="gcc",
    )
