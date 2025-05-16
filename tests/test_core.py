# Tests for the main functionalities of the core module.
# Not contain tests for advanced features like parallelism, beam search, heuristics, etc.

import os
import shutil

import stim

import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

from tesseract_decoder.core import (
    TESSERACT_PATH,
    decode_from_circuit_file,
    decode_from_detection_events,
)


def test_main_functionality():
    """
    Tests the core functionalities previously in the __main__ block of core.py,
    including both file output and stdout output.
    """
    print(f"Using Tesseract executable at: {TESSERACT_PATH}")

    # Setup a temporary directory for test files
    test_temp_dir = "tests/temp"
    if os.path.exists(test_temp_dir):
        shutil.rmtree(test_temp_dir)
    os.makedirs(test_temp_dir, exist_ok=True)

    # Files for testing
    circuit_file_name = os.path.join(test_temp_dir, "surface_code_d3_noise.stim")
    dem_file_name = os.path.join(test_temp_dir, "surface_code_d3_noise.dem")
    events_file_name = os.path.join(test_temp_dir, "events.01")
    obs_file_name = os.path.join(test_temp_dir, "obs.01")
    decoding_circuit_output_file_name = os.path.join(
        test_temp_dir, "decoded_circuit.01"
    )
    decoding_events_output_file_name = os.path.join(test_temp_dir, "decoded_events.txt")
    decoding_events_obs_output_file_name = os.path.join(
        test_temp_dir, "decoded_events_obs.txt"
    )

    # Generate d=3 surface code circuit with code-capacity noise
    surface_code_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=1,
        before_round_data_depolarization=0.2,
    )
    with open(circuit_file_name, "w") as f:
        f.write(str(surface_code_circuit))
    print(f"Generated Stim circuit saved to: {circuit_file_name}")
    assert os.path.exists(circuit_file_name)

    # Create a basic DEM from the circuit for the second example.
    dem = surface_code_circuit.detector_error_model(decompose_errors=True)
    with open(dem_file_name, "w") as f:
        f.write(str(dem))
    print(f"Generated DEM saved to: {dem_file_name}")
    assert os.path.exists(dem_file_name)

    # Arbitrary events file for testing
    dem_sampler = dem.compile_sampler()
    events, obs, _ = dem_sampler.sample(shots=10)

    with open(events_file_name, "w") as f:
        for row in events:
            f.write("".join(str(int(x)) for x in row) + "\n")
    print(f"Generated events saved to: {events_file_name}")
    assert os.path.exists(events_file_name)

    with open(obs_file_name, "w") as f:
        for row in obs:
            f.write("".join(str(int(x)) for x in row) + "\n")
    print(f"Generated obs saved to: {obs_file_name}")
    assert os.path.exists(obs_file_name)

    print("\n--- Example: Decoding from a circuit file (output to file) ---")
    circuit_result_file = decode_from_circuit_file(
        circuit_file_name,
        sample_num_shots=10,
        print_stats=True,
        out_file=decoding_circuit_output_file_name,
    )
    print(f"Return Code: {circuit_result_file['returncode']}")
    print(f"Stdout:\n{circuit_result_file['stdout']}")
    if circuit_result_file["stderr"]:
        print(f"Stderr:\n{circuit_result_file['stderr']}")
    assert circuit_result_file["returncode"] == 0
    assert os.path.exists(decoding_circuit_output_file_name)
    print(f"Output file '{decoding_circuit_output_file_name}' created.")

    print("\n--- Example: Decoding from a circuit file (output to stdout) ---")
    circuit_result_stdout = decode_from_circuit_file(
        circuit_file_name,
        sample_num_shots=10,
        print_stats=True,
        out_file=None,  # This will default to "-" internally
    )
    print(f"Return Code: {circuit_result_stdout['returncode']}")
    print(f"Stdout from stdout test:\n{circuit_result_stdout['stdout']}")
    if circuit_result_stdout["stderr"]:
        print(f"Stderr from stdout test:\n{circuit_result_stdout['stderr']}")
    assert circuit_result_stdout["returncode"] == 0
    assert circuit_result_stdout["stdout"]  # Check stdout is not empty

    print("\n--- Example: Decoding from detection events (output to file) ---")
    events_result_file = decode_from_detection_events(
        dem_file_name,
        events_file_name,
        out_file=decoding_events_output_file_name,
        print_stats=True,
    )
    print(f"Return Code: {events_result_file['returncode']}")
    print(f"Stdout:\n{events_result_file['stdout']}")
    if events_result_file["stderr"]:
        print(f"Stderr:\n{events_result_file['stderr']}")
    assert events_result_file["returncode"] == 0
    assert os.path.exists(decoding_events_output_file_name)
    print(f"Output file '{decoding_events_output_file_name}' created.")

    print("\n--- Example: Decoding from detection events (output to stdout) ---")
    events_result_stdout = decode_from_detection_events(
        dem_file_name,
        events_file_name,
        out_file=None,  # This will default to "-" internally
        print_stats=True,
    )
    print(f"Return Code: {events_result_stdout['returncode']}")
    print(f"Stdout from stdout test:\n{events_result_stdout['stdout']}")
    if events_result_stdout["stderr"]:
        print(f"Stderr from stdout test:\n{events_result_stdout['stderr']}")
    assert events_result_stdout["returncode"] == 0
    assert events_result_stdout["stdout"]  # Check stdout is not empty

    print(
        "\n--- Example: Decoding from detection events and observable flips (output to file) ---"
    )
    events_obs_result_file = decode_from_detection_events(
        dem_file_name,
        events_file_name,
        obs_file_name,
        out_file=decoding_events_obs_output_file_name,
        print_stats=True,
    )
    print(f"Return Code: {events_obs_result_file['returncode']}")
    print(f"Stdout:\n{events_obs_result_file['stdout']}")
    if events_obs_result_file["stderr"]:
        print(f"Stderr:\n{events_obs_result_file['stderr']}")
    assert events_obs_result_file["returncode"] == 0
    assert os.path.exists(decoding_events_obs_output_file_name)
    print(f"Output file '{decoding_events_obs_output_file_name}' created.")

    print(
        "\n--- Example: Decoding from detection events and observable flips (output to stdout) ---"
    )
    events_obs_result_stdout = decode_from_detection_events(
        dem_file_name,
        events_file_name,
        obs_file_name,
        out_file=None,  # This will default to "-" internally
        print_stats=True,
    )
    print(f"Return Code: {events_obs_result_stdout['returncode']}")
    print(f"Stdout from stdout test:\n{events_obs_result_stdout['stdout']}")
    if events_obs_result_stdout["stderr"]:
        print(f"Stderr from stdout test:\n{events_obs_result_stdout['stderr']}")
    assert events_obs_result_stdout["returncode"] == 0
    assert events_obs_result_stdout["stdout"]  # Check stdout is not empty

    # Clean up the temporary directory
    shutil.rmtree(test_temp_dir)
    print(f"Cleaned up temporary directory: {test_temp_dir}")


if __name__ == "__main__":
    test_main_functionality()
