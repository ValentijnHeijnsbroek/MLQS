import pandas as pd
import numpy as np
import scipy.spatial.distance as dist


def normalize_dataset(data_table, cols):
    """Normalize the specified columns to [0, 1] range."""
    dt_norm = data_table.copy()
    for col in cols:
        min_val = dt_norm[col].min()
        max_val = dt_norm[col].max()
        if max_val - min_val > 0:
            dt_norm[col] = (dt_norm[col] - min_val) / (max_val - min_val)
        else:
            dt_norm[col] = 0
    return dt_norm


def distance(data_table, d_function):
    """Compute pairwise distances between rows using the given distance function."""
    return dist.pdist(data_table.values.astype(float), metric=d_function)
