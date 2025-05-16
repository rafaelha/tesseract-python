import os
import shutil
import sys
from pathlib import Path

import numpy as np
import stim

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from tesseract_decoder.decoder import TesseractDecoder


def setup_test_environment() -> dict:
    """
    Sets up a temporary directory and generates a stim circuit and DEM for testing.

    Returns
    -------
    dict
        A dictionary containing paths to the circuit file, DEM file,
        the stim.Circuit object, and the stim.DetectorErrorModel object.
    """
    test_temp_dir = "tests/temp_decoder_test"
    if os.path.exists(test_temp_dir):
        shutil.rmtree(test_temp_dir)
    os.makedirs(test_temp_dir, exist_ok=True)

    circuit_file_name = os.path.join(test_temp_dir, "surface_code_d3_test.stim")

    # Generate d=3 surface code circuit
    surface_code_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,  # Using a few rounds for more interesting DEM
        before_round_data_depolarization=0.01,  # Small noise for testing
    )
    with open(circuit_file_name, "w") as f:
        f.write(str(surface_code_circuit))

    dem = surface_code_circuit.detector_error_model(decompose_errors=True)
    dem_file_name = os.path.join(test_temp_dir, "surface_code_d3_test.dem")
    with open(dem_file_name, "w") as f:
        f.write(str(dem))

    return {
        "temp_dir": test_temp_dir,
        "circuit_file": circuit_file_name,
        "dem_file": dem_file_name,
        "circuit": surface_code_circuit,
        "dem": dem,
    }


def cleanup_test_environment(test_temp_dir: str):
    """
    Removes the temporary directory created for testing.

    Parameters
    ----------
    test_temp_dir : str
        Path to the temporary directory to remove.
    """
    if os.path.exists(test_temp_dir):
        shutil.rmtree(test_temp_dir)


def test_tesseract_decoder_initialization():
    """
    Tests the initialization of the TesseractDecoder with a circuit and with a DEM.
    """
    env = setup_test_environment()
    circuit = env["circuit"]
    dem = env["dem"]

    # Test initialization with circuit
    try:
        decoder_from_circuit = TesseractDecoder(circuit=circuit)
        assert decoder_from_circuit.circuit is circuit
        assert decoder_from_circuit.dem is not None
        print("TesseractDecoder initialized successfully with stim.Circuit.")
    except Exception as e:
        assert False, f"Initialization with circuit failed: {e}"

    # Test initialization with DEM
    try:
        decoder_from_dem = TesseractDecoder(dem=dem)
        assert decoder_from_dem.dem is dem
        assert decoder_from_dem.circuit is None
        print("TesseractDecoder initialized successfully with stim.DetectorErrorModel.")
    except Exception as e:
        assert False, f"Initialization with DEM failed: {e}"

    # Test initialization with invalid parameters (both circuit and DEM)
    try:
        TesseractDecoder(circuit=circuit, dem=dem)
        assert False, "ValueError not raised when both circuit and DEM are provided."
    except ValueError:
        print("ValueError correctly raised when both circuit and DEM are provided.")
        pass  # Expected

    # Test initialization with invalid parameters (neither circuit nor DEM)
    try:
        TesseractDecoder()
        assert False, "ValueError not raised when neither circuit nor DEM is provided."
    except ValueError:
        print("ValueError correctly raised when neither circuit nor DEM is provided.")
        pass  # Expected

    cleanup_test_environment(env["temp_dir"])


def test_tesseract_decoder_decode_and_decode_batch():
    """
    Tests the decode and decode_batch methods of the TesseractDecoder.
    Uses a rotated surface code example.
    """
    env = setup_test_environment()
    circuit = env["circuit"]
    dem = env["dem"]
    num_observables = dem.num_observables

    # Initialize decoder
    # Using the DEM directly as it's slightly more direct for sampling
    decoder = TesseractDecoder(dem=dem)  # Using a small beam for faster tests
    print(f"Decoder initialized: {decoder!r}")

    # Generate sample detection events
    sampler = dem.compile_sampler()
    num_samples_single = 1
    num_samples_batch = 5

    # For decode method (single sample)
    det_events_single, actual_obs_single, _ = sampler.sample(shots=num_samples_single)
    # det_events_single is (1, num_detectors), actual_obs_single is (1, num_observables)

    # For decode_batch method (multiple samples)
    det_events_batch, actual_obs_batch, _ = sampler.sample(shots=num_samples_batch)
    # det_events_batch is (num_samples_batch, num_detectors)

    print(
        f"Generated {num_samples_single} sample for decode(): shape {det_events_single.shape}"
    )
    print(
        f"Generated {num_samples_batch} samples for decode_batch(): shape {det_events_batch.shape}"
    )

    # --- Test decode method ---
    print("\n--- Testing decode() method ---")
    try:
        # Input can be 1D array-like
        prediction_single = decoder.decode(
            det_events_single[0], verbose=False, print_stats=False
        )
        assert isinstance(
            prediction_single, np.ndarray
        ), "Prediction (single) should be a NumPy array"
        assert prediction_single.ndim == 1, "Prediction (single) should be 1D"
        assert (
            prediction_single.shape[0] == num_observables
        ), f"Prediction (single) shape mismatch. Expected ({num_observables},), got {prediction_single.shape}"
        assert (
            prediction_single.dtype == bool
        ), "Prediction (single) dtype should be bool"
        print(
            f"decode() successful. Input shape: {det_events_single[0].shape}, Output shape: {prediction_single.shape}"
        )

        # Test with list input
        prediction_single_list_input = decoder.decode(
            det_events_single[0].tolist(), verbose=False, print_stats=False
        )
        assert np.array_equal(
            prediction_single, prediction_single_list_input
        ), "Prediction should be same for ndarray and list input"
        print("decode() with list input successful.")

    except Exception as e:
        assert False, f"decoder.decode() failed: {e}"

    # --- Test decode_batch method ---
    print("\n--- Testing decode_batch() method ---")
    try:
        prediction_batch = decoder.decode_batch(
            det_events_batch, verbose=False, print_stats=False
        )
        assert isinstance(
            prediction_batch, np.ndarray
        ), "Prediction (batch) should be a NumPy array"
        assert prediction_batch.ndim == 2, "Prediction (batch) should be 2D"
        assert prediction_batch.shape == (
            num_samples_batch,
            num_observables,
        ), f"Prediction (batch) shape mismatch. Expected ({num_samples_batch}, {num_observables}), got {prediction_batch.shape}"
        assert prediction_batch.dtype == bool, "Prediction (batch) dtype should be bool"
        print(
            f"decode_batch() successful. Input shape: {det_events_batch.shape}, Output shape: {prediction_batch.shape}"
        )

        # Test with list of lists input
        prediction_batch_list_input = decoder.decode_batch(
            det_events_batch.tolist(), verbose=False, print_stats=False
        )
        assert np.array_equal(
            prediction_batch, prediction_batch_list_input
        ), "Batch prediction should be same for ndarray and list of lists input"
        print("decode_batch() with list of lists input successful.")

        # Test with empty batch
        empty_batch_input = np.empty(
            (0, dem.num_detectors if dem.num_detectors > 0 else 10), dtype=bool
        )  # Ensure 2D
        if (
            dem.num_detectors == 0 and num_observables == 0
        ):  # Special case for trivial DEMs
            empty_prediction = decoder.decode_batch(empty_batch_input)
            assert empty_prediction.shape == (0, 0)  # stim 0 obs produces (0,0)
            print("decode_batch() with empty input (0 detectors, 0 obs) successful.")
        elif dem.num_detectors > 0:  # Normal case
            empty_prediction = decoder.decode_batch(empty_batch_input)
            assert empty_prediction.shape == (0, num_observables)
            print("decode_batch() with empty input successful.")
        else:  # dem.num_detectors == 0 but num_observables > 0 (e.g. from circuit with only logical obs)
            # This case is a bit tricky as Tesseract might not be able to handle 0 detectors if it expects them
            # For now, assume Tesseract handles it or the TesseractDecoder wrapper would need to.
            # Current `decode_batch` returns (0, num_observables) based on self.dem.num_observables
            print(
                f"Skipping empty batch test for 0 detectors and {num_observables} observables, as behavior might be ambiguous."
            )
            pass

    except Exception as e:
        assert False, f"decoder.decode_batch() failed: {e}"

    # --- Test verbose and print_stats flags (just ensuring they don't crash) ---
    print("\n--- Testing verbose and print_stats flags ---")
    try:
        # For decode
        _ = decoder.decode(det_events_single[0], verbose=True, print_stats=True)
        print("decode() with verbose=True, print_stats=True ran without error.")
        # For decode_batch
        _ = decoder.decode_batch(det_events_batch, verbose=True, print_stats=True)
        print("decode_batch() with verbose=True, print_stats=True ran without error.")
    except Exception as e:
        assert False, f"Decoder methods with verbose/print_stats flags failed: {e}"

    # Test __repr__
    print("\n--- Testing __repr__ ---")
    try:
        repr_str = repr(decoder)
        assert "TesseractDecoder" in repr_str
        assert (
            "dem=<stim.DetectorErrorModel>" in repr_str
        )  # since we initialized with DEM
        print(f"__repr__ output: {repr_str}")

        decoder_circuit_init = TesseractDecoder(
            circuit=circuit, beam_climbing=True, det_penalty=0.1
        )
        repr_str_circuit = repr(decoder_circuit_init)
        assert "circuit=<stim.Circuit>" in repr_str_circuit
        assert "beam_climbing=True" in repr_str_circuit
        assert "det_penalty=0.1" in repr_str_circuit
        print(f"__repr__ for circuit init: {repr_str_circuit}")

        decoder_no_params = TesseractDecoder(circuit=circuit)
        repr_str_no_params = repr(decoder_no_params)
        assert (
            "TesseractDecoder(circuit=<stim.Circuit>)" == repr_str_no_params
            or "TesseractDecoder(circuit=<stim.Circuit>, )" == repr_str_no_params
        )  # Allow for trailing comma if params are empty
        print(f"__repr__ for circuit init (no extra params): {repr_str_no_params}")

    except Exception as e:
        assert False, f"__repr__ method failed: {e}"

    cleanup_test_environment(env["temp_dir"])
    print("\nAll TesseractDecoder tests passed.")


def test_tesseract_decoder_simulate():
    """
    Tests the simulate method of the TesseractDecoder.
    """
    env = setup_test_environment()
    circuit = env["circuit"]
    dem = env["dem"]  # For testing error case
    num_observables = circuit.num_observables

    print("\n--- Testing simulate() method ---")

    # Test successful simulation
    try:
        decoder = TesseractDecoder(circuit=circuit)  # Small beam for speed
        num_shots = 3
        predictions = decoder.simulate(
            shots=num_shots, sample_seed=42, verbose=False, print_stats=False
        )

        assert isinstance(
            predictions, np.ndarray
        ), "Simulate predictions should be a NumPy array"
        assert predictions.ndim == 2, "Simulate predictions should be 2D"
        assert predictions.shape == (
            num_shots,
            num_observables,
        ), f"Simulate predictions shape mismatch. Expected ({num_shots}, {num_observables}), got {predictions.shape}"
        assert predictions.dtype == bool, "Simulate predictions dtype should be bool"
        print(f"simulate() successful. Output shape: {predictions.shape}")

        # Test verbose and print_stats flags (just ensuring they don't crash)
        _ = decoder.simulate(shots=1, verbose=True, print_stats=True)
        print("simulate() with verbose=True, print_stats=True ran without error.")

    except Exception as e:
        assert False, f"decoder.simulate() failed during normal operation: {e}"

    # Test ValueError if initialized with DEM instead of Circuit
    print("\n--- Testing simulate() error conditions ---")
    try:
        decoder_dem_init = TesseractDecoder(dem=dem)
        decoder_dem_init.simulate(shots=1)
        assert (
            False
        ), "ValueError not raised when simulate() called on DEM-initialized decoder."
    except ValueError as ve:
        assert "must be initialized with a stim.Circuit" in str(ve)
        print("ValueError correctly raised for simulate() on DEM-initialized decoder.")
    except Exception as e:
        assert (
            False
        ), f"Unexpected exception for simulate() on DEM-initialized decoder: {e}"

    # Test ValueError for non-positive shots
    try:
        decoder_circuit_init = TesseractDecoder(circuit=circuit)
        decoder_circuit_init.simulate(shots=0)
        assert False, "ValueError not raised when simulate() called with shots=0."
    except ValueError as ve:
        assert "must be positive" in str(ve).lower()
        print("ValueError correctly raised for simulate() with shots=0.")
    except Exception as e:
        assert False, f"Unexpected exception for simulate() with shots=0: {e}"

    try:
        decoder_circuit_init = TesseractDecoder(circuit=circuit)
        decoder_circuit_init.simulate(shots=-1)
        assert False, "ValueError not raised when simulate() called with shots=-1."
    except ValueError as ve:
        assert "must be positive" in str(ve).lower()
        print("ValueError correctly raised for simulate() with shots=-1.")
    except Exception as e:
        assert False, f"Unexpected exception for simulate() with shots=-1: {e}"

    cleanup_test_environment(env["temp_dir"])
    print("\nAll TesseractDecoder simulate tests passed.")


if __name__ == "__main__":
    test_tesseract_decoder_initialization()
    test_tesseract_decoder_decode_and_decode_batch()
    test_tesseract_decoder_simulate()
