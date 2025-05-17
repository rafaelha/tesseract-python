import sys
from pathlib import Path

import numpy as np

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from tesseract_decoder.utils import convert_array_to_01_format, convert_01_format_to_array


def test_convert_array_to_01_format():
    array = np.array([[True, False], [False, True]])
    expected = "10\n01\n"
    result = convert_array_to_01_format(array)
    assert result == expected


def test_convert_01_format_to_array():
    s = "10\n01\n"
    expected = np.array([[True, False], [False, True]])
    result = convert_01_format_to_array(s)
    assert np.array_equal(result, expected)
    assert result.dtype == bool


def test_round_trip_conversion():
    original = np.array([[False, True, True], [True, False, False]])
    converted = convert_array_to_01_format(original)
    recovered = convert_01_format_to_array(converted)
    assert np.array_equal(recovered, original)
    assert recovered.dtype == bool
