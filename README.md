# Python Wrapper for Tesseract Decoder

This package provides a Python wrapper for the [Tesseract Decoder, a search-based decoder for quantum error correction](https://github.com/quantumlib/tesseract-decoder).

## Installation

You have two main options for installing the package.

### Option 1: Clone with Submodule and Build Locally

This option is recommended if you want to ensure the bundled C++ Tesseract decoder is compiled and used directly.

1.  **Clone the repository with submodules:**
    ```bash
    git clone --recurse-submodules https://github.com/seokhyung-lee/tesseract-python
    cd tesseract-python
    ```

2.  **Build the Tesseract C++ decoder:**
    Navigate to the submodule directory and build using Bazel.
    ```bash
    cd src/tesseract_decoder/tesseract-decoder
    bazel build src:all
    cd ../../.. 
    ```
    This will create the `tesseract` executable in `src/tesseract_decoder/tesseract-decoder/bazel-bin/src/`. The Python wrapper is configured to find it there.

3.  **Run tests (Optional):**
    You can run the Python wrapper's test suite to ensure everything is working correctly. Make sure you have `pytest` installed (`pip install pytest`).
    ```bash
    pytest tests
    ```

4.  **Install the Python wrapper:**
    ```bash
    pip install .
    ```
    Or, for an editable install:
    ```bash
    pip install -e .
    ```

### Option 2: Use a Pre-built Tesseract Executable

If you have already built or installed the Tesseract C++ decoder from its [original repository](https://github.com/quantumlib/tesseract-decoder) and added the `tesseract` executable to your system's `PATH`, you can install this Python wrapper without the submodule.

1.  **Ensure `tesseract` is in your PATH:**
    Verify that you can run `tesseract --version` from your terminal. If not, please refer to the [original Tesseract decoder documentation](https://github.com/quantumlib/tesseract-decoder?tab=readme-ov-file#installation) for build and installation instructions, and ensure the built executable is accessible in your `PATH`.

2.  **Install the Python wrapper:**
    ```bash
    pip install git+https://github.com/seokhyung-lee/tesseract-python.git
    ```

    The Python wrapper will then attempt to find the `tesseract` executable in your system `PATH`.

## Usage

Currently only supports decoding from circuit or DEM files as the original command-line
interface. I will add more flexible ways soon.

See the [original Tesseract decoder documentation](https://github.com/quantumlib/tesseract-decoder?tab=readme-ov-file#installation) for details on the parameters of the following functions.

**1. Decoding from a Stim circuit file:**

```python
from tesseract_decoder import decode_from_circuit_file

# Sample 1000 shots from a circuit and save predictions
results = decode_from_circuit_file(
    "circuit.stim",
    sample_num_shots=1000,
    out_file="predictions.01",  # if None, output to stdout
)
print(f"Stdout: {results['stdout']}")

# Example with advanced options
advanced_results = decode_from_circuit_file(
    "circuit.stim",
    sample_num_shots=10000,
    pq_limit=1000000,
    at_most_two_errors_per_detector=True,
    det_order_seed=232852747,
    sample_seed=232856747,
    threads=32,
    print_stats=True,
    beam=23,
    num_det_orders=1,
    shot_range_begin=582,
    shot_range_end=583
)
print(f"Advanced Stdout: {advanced_results['stdout']}")
```

**2. Decoding from pre-existing detection events and observable flips:**

```python
from tesseract_decoder import decode_from_detection_events

# Decode from a detection event file using a DEM
results_no_obs = decode_from_detection_events(
    "detector_error_model.dem",
    "detection_events.01",
    out_file="decoded_no_obs.txt"
)
print(f"Stdout: {results_no_obs['stdout']}")

# Decode from a detection event file and observable flips file using a DEM
results_with_obs = decode_from_detection_events(
    "detector_error_model.dem",
    "detection_events.01",
    "obs.01",
    out_file="decoded_with_obs.txt"
)
print(f"Stdout: {results_with_obs['stdout']}")
```

## Original Project

For more information on the underlying C++ Tesseract decoder, please see the [original repository](https://github.com/quantumlib/tesseract-decoder). 