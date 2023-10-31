from enum import Enum

import numpy as np


class ZeroSumNorm(Enum):
    NONE = 'NONE'
    LINEAR = 'LINEAR'

def apply_zero_sum_norm(
        values: np.ndarray,  # array of shape (2,)
        zero_sum_norm: ZeroSumNorm,
) -> np.ndarray:
    if (len(values.shape) != 1 or values.shape[0] != 2) and zero_sum_norm != ZeroSumNorm.NONE \
            and zero_sum_norm != ZeroSumNorm.NONE.value:
        raise ValueError(f"Invalid array shape for zero sum norm: {values.shape}")
    if zero_sum_norm == ZeroSumNorm.NONE or zero_sum_norm == ZeroSumNorm.NONE.value:
        return values
    elif zero_sum_norm == ZeroSumNorm.LINEAR or zero_sum_norm == ZeroSumNorm.LINEAR.value:
        avg = np.average(values)
        norm = values - avg
        return norm
    else:
        raise ValueError(f"Invalid norm type: {zero_sum_norm}")
