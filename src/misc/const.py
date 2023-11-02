import math

import seaborn

# the constant of the golden ratio phi
PHI: float = (1 + math.sqrt(5)) / 2

_color_indices = [0, 1, 2, 4, 7, 9, 3, 5, 6, 8]
COLORS = [list(seaborn.color_palette('colorblind', n_colors=10))[i] for i in _color_indices]
LIGHT_COLORS = [list(seaborn.color_palette('pastel', n_colors=10))[i] for i in _color_indices]
LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'solid', 'solid', 'solid']
