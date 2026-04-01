from enum import Enum
import math

import torch
import torch.ao.quantization as tq


class PowerOfTwoObserver(tq.MinMaxObserver):
    """
    Observer module for power-of-two quantization (dyadic quantization with b = 1).
    """

    def scale_approximate(self, scale: float, max_shift_amount=8) -> float:
        c = round(min(max(-math.log2(scale), 0), max_shift_amount))
        return 2**-c

    def calculate_qparams(self):
        """Calculates the quantization parameters with scale as power of two."""
        min_val, max_val = self.min_val.item(), self.max_val.item()

        """ Calculate zero_point as in the base class """
        if self.qscheme == torch.per_tensor_symmetric:
            max_range = max(abs(min_val), abs(max_val))
            scale = (2 * max_range) / ((2**8) - 1)
            if self.dtype == torch.qint8:
                zero_point = 0
            else:
                zero_point = 128
        else:
            max_range = max_val - min_val
            scale = max_range / ((2**8) - 1)
            zero_point = -round(min_val / scale)

        scale = self.scale_approximate(scale)
        scale = torch.tensor(scale, dtype=torch.float32)
        zero_point = torch.tensor(zero_point, dtype=torch.int64)
        return scale, zero_point

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}, scale=PowerOfTwo"


class DyadicObserver(tq.MinMaxObserver):
    """
    Observer module for dyadic quantization.
    """

    def scale_to_dyadic(self, scale, b_width=8, c_width=4):
        max_shift_ammount = 2**c_width - 1
        sign = 1 if scale >= 0 else -1
        scale = abs(scale)
        fp_val = scale * 2**max_shift_ammount

        dyadic_c = int(max_shift_ammount)
        dyadic_b = int(round(fp_val))

        """ check if b is not overflow """
        while dyadic_b > 2**b_width - 1:
            if dyadic_c == 0:
                dyadic_b = 2**b_width - 1
                break
            dyadic_c -= 1
            dyadic_b //= 2

        return dyadic_b / (2**dyadic_c) * sign

    def calculate_qparams(self):
        """Calculates the quantization parameters with scale as dyadic number."""
        min_val, max_val = self.min_val.item(), self.max_val.item()

        """ Calculate zero_point as in the base class """
        if self.qscheme == torch.per_tensor_symmetric:
            max_range = max(abs(min_val), abs(max_val))
            scale = (2 * max_range) / ((2**8) - 1)
            if self.dtype == torch.qint8:
                zero_point = 0
            else:
                zero_point = 128
        else:
            max_range = max_val - min_val
            scale = max_range / ((2**8) - 1)
            zero_point = -round(min_val / scale)

        scale = self.scale_to_dyadic(scale, b_width=8, c_width=4)
        scale = torch.tensor(scale, dtype=torch.float32)
        zero_point = torch.tensor(zero_point, dtype=torch.int64)
        return scale, zero_point

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}, scale=Dyadic"


class CustomQConfig(Enum):
    POWER2 = tq.QConfig(
        activation=PowerOfTwoObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric
        ),
        weight=PowerOfTwoObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    DYADIC = tq.QConfig(
        activation=DyadicObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric
        ),
        weight=DyadicObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    QNNPACK = tq.get_default_qconfig("qnnpack")
    DEFAULT = None
