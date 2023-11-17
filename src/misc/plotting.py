from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt


def plot_filled_std_curves(
        x: np.ndarray,
        mean: np.ndarray,
        color: Any,
        lighter_color: Any,
        std: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        lower: Optional[np.ndarray] = None,
        linestyle: str = '-',
        marker: Optional[str] = None,
        label: Optional[str] = None,
        alpha: float = 0.2,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
):
    if (upper is None) != (lower is None):
        raise ValueError(f"Need to specify both upper and lower")
    if (std is None) == (upper is None):
        raise ValueError(f"Need to specify either std or upper/lower")
    if std is not None:
        upper = mean + std
        lower = mean - std
    if min_val is not None and lower is not None and upper is not None:
        mean = np.maximum(mean, min_val)
        lower = np.maximum(lower, min_val)
        upper = np.maximum(upper, min_val)
    if max_val is not None and lower is not None and upper is not None:
        mean = np.minimum(mean, max_val)
        upper = np.minimum(upper, max_val)
        lower = np.minimum(lower, max_val)
    if upper is None or lower is None:
        raise Exception("This should never happen")
    plt.plot(x, upper, color=lighter_color, alpha=alpha)
    plt.plot(x, lower, color=lighter_color, alpha=alpha)
    plt.fill_between(x, lower, upper, color=lighter_color, alpha=alpha)
    plt.plot(x, mean, color=color, label=label, linestyle=linestyle, marker=marker, markersize=4)
