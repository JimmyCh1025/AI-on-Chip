# Python/build_model.py
from utils import *
import logging
import argparse
import os


def main(opts):
    # Load ONNX model
    # transforms into TVM relay model
    relay_model, params = load_onnx_model(model_path=opts.model_path)

    os.makedirs(opts.log_dir, exist_ok=True)

    if opts.dump_origin_only:
        dump_relay_model(relay_model, f"{opts.log_dir}/relay_mc_origin_model.txt")
        print("Dump origin Relay model Done")
        return

    # PASS merge_composite_pass
    # 1. Op fused
    # 2. Target annotate
    progress, fused_model = merge_composite_pass(relay_model)

    # Dump relay model
    for k, p in progress.items():
        dump_relay_model(p, f"{opts.log_dir}/relay_mc_{k}_model.txt")

    if opts.dump_relay_only:
        print("Dump Relay models Done")
        return

    # Build model
    # codegen
    build_model(fused_model, params, opts)
    print("Build model Done")

    # gen test data
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_gen = Dataset_Generator(
        datasets.CIFAR10,
        root=f"{opts.dataset_dir}/cifar10",
        eval_transform=eval_transform,
    )
    dataset_gen.gen_bin(f"{opts.out_dir}/input.bin", 10)
    print("Dump dataset Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("-o", "--out-dir", default="./output")
    parser.add_argument("-l", "--log-dir", default="./log")
    parser.add_argument("-d", "--dataset_dir", default="./data")
    parser.add_argument("-t", "--test", action="store_true")
    dump_group = parser.add_mutually_exclusive_group()
    dump_group.add_argument("--dump-origin-only", action="store_true")
    dump_group.add_argument("--dump-relay-only", action="store_true")
    opts = parser.parse_args()

    main(opts)
