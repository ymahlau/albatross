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
            norm = values - avg
        elif len(values.shape) == 2:
            norm = values - avg[:, np.newaxis]
        else:
            raise ValueError(f"Invalid array shape for zero sum norm: {values.shape}")
        return norm
    elif norm == UtilityNorm.FULL_COOP:
        avg = np.average(values, axis=-1)
        if len(values.shape) == 1:
            norm = np.ones_like(values) * avg
        elif len(values.shape) == 2:
            norm = np.ones_like(values) * avg[:, np.newaxis]
        return norm
    else:
        raise ValueError(f"Invalid norm type: {norm}")
