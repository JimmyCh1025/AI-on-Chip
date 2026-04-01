import torch


def torch2onnx(model, output_file_path, dummy_input):
    export_kwargs = dict(
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_file_path,
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            output_file_path,
            **export_kwargs,
        )

    print(f"Model saved as {output_file_path}")
