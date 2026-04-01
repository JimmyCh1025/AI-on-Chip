from abc import ABC
from dataclasses import dataclass, asdict


class ShapeParam(ABC):
    def to_dict(self) -> dict:
        """Convert the object's attributes to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Create an instance of the class from a dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class Conv2DShapeParam(ShapeParam):
    """Follow the notation in the Eyeriss paper"""

    N: int  # batch size
    H: int  # input height
    W: int  # input width
    R: int  # filter height
    S: int  # filter width
    E: int  # output height
    F: int  # output width
    C: int  # input channels
    M: int  # output channels
    U: int = 1  # stride
    P: int = 1  # padding


@dataclass(frozen=True)
class LinearShapeParam(ShapeParam):
    N: int  # batch size
    in_features: int
    out_features: int


@dataclass(frozen=True)
class MaxPool2DShapeParam(ShapeParam):
    N: int  # batch size
    kernel_size: int
    stride: int
