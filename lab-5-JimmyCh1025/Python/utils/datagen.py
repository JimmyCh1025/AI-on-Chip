import torch
import struct
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset_Generator(object):
    def __init__(self, source, root="data", eval_transform=None):
        self.root = root
        self.eval_transform = eval_transform
        self.test_dataset = source(
            root=self.root,
            train=False,
            download=True,
            transform=self.eval_transform,
        )
        self.testloader = DataLoader(self.test_dataset, batch_size=1, num_workers=1, shuffle=False)
        self.classes = []

    def fetch_data(self, num_data_per_class):
        self.classes = self.test_dataset.classes
        data_dict = dict()

        for idx, c in enumerate(self.classes):
            data_dict[c] = []
            for img, y in self.testloader:
                if idx == y:
                    data_dict[c].append(img)
                if len(data_dict[c]) >= num_data_per_class:
                    break
        return data_dict

    def gen_bin(self, output_path, num_data_per_class=10):
        data_dict = self.fetch_data(num_data_per_class=num_data_per_class)

        with open(output_path, "wb") as f:
            num_classes = len(self.classes)
            f.write(struct.pack("I", num_classes))  # Total number of classes

            for class_name in self.classes:
                encoded_name = class_name.encode("utf-8")
                name_length = len(encoded_name)
                f.write(struct.pack("I", name_length))
                f.write(encoded_name)

            first_data_shape = list(data_dict.values())[0][0].shape
            flattened_size = np.prod(first_data_shape)
            f.write(struct.pack("I", num_data_per_class))
            f.write(struct.pack("I", flattened_size))

            for values in data_dict.values():
                for value in values:
                    np_array = value.numpy().astype("float32")
                    f.write(np_array.tobytes())
