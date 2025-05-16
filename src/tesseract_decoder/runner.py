import os
import subprocess
import sys
from importlib.resources import files


def main() -> None:
    bin_path = files("tesseract_decoder").joinpath("_bin", "tesseract")
    if os.name == "nt" and not bin_path.suffix:
        bin_path = bin_path.with_suffix(".exe")

    os.execv(bin_path, [str(bin_path), *sys.argv[1:]])
