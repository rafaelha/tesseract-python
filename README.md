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
    ```

3. **Copy the Tesseract executable:**
    ```bash
    cp bazel-bin/src/tesseract ../_bin/tesseract
    chmod +x ../_bin/tesseract
    cd ../../.. 
    ```

4.  **Run tests (Optional):**
    You can run the Python wrapper's test suite to ensure the core function is working correctly. (It doesn't include tests for advanced features.) Make sure you have `pytest` installed (`pip install pytest`).
    ```bash
    pytest tests
    ```

5.  **Install the Python wrapper:**
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
    Verify that you can run `tesseract --version` from your terminal. If not, please refer to the [original Tesseract decoder documentation](https://github.com/quantumlib/tesseract-decoder?tab=readme-ov-file#installation) for build and installation instructions, and ensure the built executable is accessible in your `PATH`. Note that you can find the tesseract executable at `tesseract-decoder/bazel-bin/src/tesseract` after compiling.

2.  **Install the Python wrapper:**
    ```bash
    pip install git+https://github.com/seokhyung-lee/tesseract-python.git
    ```

    The Python wrapper will then attempt to find the `tesseract` executable in your system `PATH`.

## Usage

This package offers two main ways to interact with the Tesseract decoder:
1.  **The `TesseractDecoder` Class**
2.  **Direct Function Calls (`decode_from_circuit_file`, `decode_from_detection_events`)**


See the docstrings of the functions/class or the [original Tesseract decoder documentation](https://github.com/quantumlib/tesseract-decoder?tab=readme-ov-file#command-line-interface) for details on all available parameters.

### 1. Using the `TesseractDecoder` Class

The `TesseractDecoder` class allows for integrated usage within Python, working directly with `stim` objects.

```python
import stim
from tesseract_decoder import TesseractDecoder

# Assume that a stim.Circuit `circuit` is predefined
decoder = TesseractDecoder(circuit=circuit)

# Decode detection events from a single sample
# det_events_sng_sample: 1D array-like with length num_detectors
preds = decoder.decode(det_events_sng_sample)

# Batch decoding
# det_events_multiple_samples: 2D array-like with shape (shots, num_detectors)
preds = decoder.decode_batch(det_events_multiple_samples)

# Simulate the circuit for a given number of shots
results = decoder.simulate(100)
```

### 2. Direct Function Calls

These functions work similarly as the commands in the original Tesseract software.

**1.1. Decoding from a Stim circuit file:**

```python
from tesseract_decoder import decode_from_circuit_file

# Sample 1000 shots from a circuit file and save predictions to another file
results = decode_from_circuit_file(
    circuit_file="circuit.stim",
    sample_num_shots=1000,
    out_file="predictions.01",    # If None, output is in results['stdout']
)

# Example with advanced options
advanced_results = decode_from_circuit_file(
    circuit_file="circuit.stim",
    sample_num_shots=10000,
    pq_limit=1000000,
    at_most_two_errors_per_detector=True,
    det_order_seed=232852747,
    sample_seed=232856747,
    threads=32,
    print_stats=True, # Stream Tesseract's stats to console
    beam=23,
    num_det_orders=1,
    shot_range_begin=582,
    shot_range_end=583
)
# The 'results' dictionary contains 'stdout', 'stderr', and 'returncode'.
```

**1.2. Decoding from pre-existing detection events and observable flips:**

```python
from tesseract_decoder import decode_from_detection_events

# Decode from a detection event file using a DEM file
results_no_obs = decode_from_detection_events(
    dem_file="detector_error_model.dem",
    detection_event_file="detection_events.01",
    out_file="decoded_no_obs.txt"
)

# Decode using a DEM, detection events, and an observable flips file
results_with_obs = decode_from_detection_events(
    dem_file="detector_error_model.dem",
    detection_event_file="detection_events.01",
    obs_in_file="obs.01",
    out_file="decoded_with_obs.txt",
)
```

## Original Project

For more information on the underlying C++ Tesseract decoder, please see the [original repository](https://github.com/quantumlib/tesseract-decoder). 