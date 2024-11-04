from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MultipleLocator
import seaborn as sns


def aesthetic_2D():
    plt.rcParams.update({
        # Matplotlib style settings similar to seaborn's default style
        "axes.facecolor": "#eaeaf2",
        "axes.edgecolor": "white",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1,
        "axes.axisbelow": True,
        "xtick.color": "gray",
        "ytick.color": "gray",

        # Additional stylistic settings
        "figure.facecolor": "white",
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.fancybox": True,
        "legend.edgecolor": 'lightgray',
    })
    