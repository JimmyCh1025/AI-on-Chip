import argparse
import onnxruntime as rt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_path", type=str, help="path to the ONNX model")
    return parser.parse_args()


def main():
    args = parse_args()
    session = rt.InferenceSession(args.model_path)
    input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    result = session.run(None, {"input": input_data})[0]
    print(result.shape)
    print(result)


if __name__ == "__main__":
    main()
