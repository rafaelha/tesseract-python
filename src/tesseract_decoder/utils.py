from typing import List, Tuple

import numpy as np


def convert_array_to_01_format(array: List[List[bool | int]] | np.ndarray) -> str:
    """
    Convert a 2D array to a "01" format string.

    Parameters
    ----------
    array : 2D array-like of int or bool
        2D array to convert.

    Returns
    -------
    str
        Converted "01" format string.
    """
    return "\n".join("".join(str(int(x)) for x in row) for row in array) + "\n"


def convert_01_format_to_array(s: str) -> np.ndarray:
    """
    Convert a "01" format string to a 2D numpy array.

    Parameters
    ----------
    s : str
        "01" format string to convert.

    Returns
    -------
    numpy 2D array
        Converted 2D numpy array of bools.
    """
    lines = s.strip().split("\n")
    array_list = []
    for line in lines:
        if line:  # Ensure the line is not empty
            array_list.append([int(char) for char in line])

    return np.array(array_list, dtype=bool)
