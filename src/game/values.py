from enum import Enum

import numpy as np


class UtilityNorm(Enum):
    NONE = 'NONE'
    ZERO_SUM = 'ZERO_SUM'
    FULL_COOP = 'FULL_COOP'


def apply_utility_norm(
        values: np.ndarray,
        norm: UtilityNorm,
) -> np.ndarray:
    if norm == UtilityNorm.NONE:
        return values
    elif norm == UtilityNorm.ZERO_SUM:
        if values.shape[-1] != 2:
            raise ValueError(f"Invalid array shape for zero sum norm: {values.shape}")
        avg = np.average(values, axis=-1)
        if len(values.shape) == 1:
            norm_arr = values - avg
        elif len(values.shape) == 2:
            norm_arr = values - avg[:, np.newaxis]
        else:
            raise ValueError(f"Invalid array shape for zero sum norm: {values.shape}")
        return norm_arr
    elif norm == UtilityNorm.FULL_COOP:
        avg = np.average(values, axis=-1)
        if len(values.shape) == 1:
            norm_arr = np.ones_like(values) * avg
        elif len(values.shape) == 2:
            norm_arr = np.ones_like(values) * avg[:, np.newaxis]
        else:
            raise ValueError(f"Invalid array shape for fully coop norm: {values.shape}")
        return norm_arr
    else:
        raise ValueError(f"Invalid norm type: {norm}")
