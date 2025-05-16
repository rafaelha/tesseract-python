import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import stim

from .core import decode_from_circuit_file, decode_from_detection_events
from .utils import convert_01_format_to_array, convert_array_to_01_format


class TesseractDecoder:
    """
    Class for the Tesseract decoder, a search-based decoder for quantum error correction.
    """

    def __init__(
        self,
        *,
        circuit: stim.Circuit | None = None,
        dem: stim.DetectorErrorModel | None = None,
        beam: int | None = None,
        beam_climbing: bool = False,
        pq_limit: int | None = None,
        num_det_orders: int | None = None,
        det_order_seed: int | None = None,
        det_penalty: float | None = None,
        at_most_two_errors_per_detector: bool = False,
        no_revisit_dets: bool = False,
        no_merge_errors: bool = False,
    ):
        """
        Initializes the TesseractDecoder with a Stim circuit or DEM object and decoding parameters.

        Exactly one of `circuit` or `dem` must be provided.

        Parameters
        ----------
        circuit : stim.Circuit, optional
            A Stim circuit object.
        dem : stim.DetectorErrorModel, optional
            A Stim detector error model object.
        beam : int, optional
            Beam to use for truncation (default = infinity).
        beam_climbing : bool, default False
            Use beam-climbing heuristic.
        pq_limit : int, optional
            Maximum size of the priority queue (default = infinity).
        num_det_orders : int, optional
            Number of ways to orient the manifold when reordering the detectors.
        det_order_seed : int, optional
            Seed used when initializing the random detector traversal orderings.
        det_penalty : float, optional
            Penalty cost to add per activated detector in the residual syndrome.
        at_most_two_errors_per_detector : bool, default False
            Use heuristic limitation of at most 2 errors per detector.
        no_revisit_dets : bool, default False
            Use no-revisit-dets heuristic.
        no_merge_errors : bool, default False
            If provided, will not merge identical error mechanisms.

        Raises
        ------
        ValueError
            If not exactly one of `circuit` or `dem` is provided.
        """
        if (circuit is None and dem is None) or (
            circuit is not None and dem is not None
        ):
            raise ValueError("Exactly one of `circuit` or `dem` must be provided.")

        self.circuit: stim.Circuit | None = circuit
        if circuit is None:
            self.dem: stim.DetectorErrorModel | None = dem
        else:
            self.dem = circuit.detector_error_model()

        # Decoding parameters (excluding threads, verbose, print_stats which are method-specific)
        self.decoding_prms = {
            "beam": beam,
            "beam_climbing": beam_climbing,
            "pq_limit": pq_limit,
            "num_det_orders": num_det_orders,
            "det_order_seed": det_order_seed,
            "det_penalty": det_penalty,
            "at_most_two_errors_per_detector": at_most_two_errors_per_detector,
            "no_revisit_dets": no_revisit_dets,
            "no_merge_errors": no_merge_errors,
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the TesseractDecoder object.

        Returns
        -------
        str
            String representation of the TesseractDecoder.
        """
        params_parts = []
        for k, v in self.decoding_prms.items():
            if v is not None:  # For bools, False is a valid value we want to show
                if isinstance(v, bool):
                    if (
                        v
                    ):  # only add if True for boolean flags to match constructor style
                        params_parts.append(f"{k}={v!r}")
                else:
                    params_parts.append(f"{k}={v!r}")
        params_str = ", ".join(params_parts)

        init_arg_str = ""
        if self.circuit is not None:
            init_arg_str = "circuit=<stim.Circuit>"
        elif self.dem is not None:
            init_arg_str = "dem=<stim.DetectorErrorModel>"
        # Else, if both are None, it's an invalid state handled by __init__,
        # but __repr__ should still be robust.

        if params_str:
            return f"TesseractDecoder({init_arg_str}, {params_str})"
        else:
            return f"TesseractDecoder({init_arg_str})"

    def _call_core_decoder(
        self,
        det_events_str: str,
        threads: int | None = None,
        verbose_flag: bool = False,
        print_stats_flag: bool = False,
    ) -> Dict[str, Any]:
        """
        Internal helper to call the core Tesseract decoder function.
        Manages temporary files for DEM, detection events, and output predictions.

        Parameters
        ----------
        det_events_str : str
            The detection events as a string in '01' format.
        threads : int, optional
            Number of decoder threads to use.
        verbose_flag : bool, default False
            Activates verbose Tesseract output for this call.
        print_stats_flag : bool, default False
            Activates Tesseract statistics printing for this call.
        """
        if self.dem is None:  # Should not happen if __init__ is called correctly
            raise ValueError("Decoder's Detector Error Model (DEM) is not initialized.")

        dem_file_path = None
        detection_event_file_path = None
        prediction_output_file_path = None
        try:
            # Create temporary DEM file
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".dem"
            ) as dem_f:
                self.dem.to_file(dem_f.name)
                dem_file_path = dem_f.name

            # Create temporary detection event file
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".01"
            ) as det_f:
                det_f.write(det_events_str)
                detection_event_file_path = det_f.name

            # Create temporary file for prediction output
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as pred_out_f:
                prediction_output_file_path = pred_out_f.name
                # We just need the name, core.decode_from_detection_events will write to it.
                # Ensure it's closed before tesseract tries to write to it, though NamedTemporaryFile(delete=False)
                # should be fine as the file is created on disk.

            # Prepare kwargs for other Tesseract options from self.decoding_prms
            core_specific_kwargs = {
                k: v
                for k, v in self.decoding_prms.items()
                if v is not None or isinstance(v, bool)
            }

            stream_output_to_console = verbose_flag or print_stats_flag

            results_core = decode_from_detection_events(
                dem_file=dem_file_path,
                detection_event_file=detection_event_file_path,
                detection_event_format="01",
                out_file=prediction_output_file_path,  # Write to temp file
                out_format="01",
                threads=threads,
                verbose=verbose_flag,
                print_stats=print_stats_flag,
                stream_output=stream_output_to_console,  # Pass stream_output flag
                # Pass other decoding parameters stored in the instance
                **core_specific_kwargs,
            )

            output_content = ""
            if (
                results_core["returncode"] == 0
                and prediction_output_file_path
                and os.path.exists(prediction_output_file_path)
            ):
                with open(prediction_output_file_path, "r") as f_out:
                    output_content = f_out.read()

            # Populate a dictionary similar to what subprocess.run would return for stdout/stderr
            results = {
                "output": output_content,  # Content from the prediction output file
                "stderr": results_core["stderr"],  # stderr from the core call (if any)
                "returncode": results_core[
                    "returncode"
                ],  # return code from the core call
            }

            return results

        finally:
            if dem_file_path and os.path.exists(dem_file_path):
                os.remove(dem_file_path)
            if detection_event_file_path and os.path.exists(detection_event_file_path):
                os.remove(detection_event_file_path)
            if prediction_output_file_path and os.path.exists(
                prediction_output_file_path
            ):
                os.remove(
                    prediction_output_file_path
                )  # Clean up prediction output file

    def decode(
        self,
        det_events: np.ndarray | List[int | bool],
        *,
        threads: int | None = None,
        verbose: bool = False,
        print_stats: bool = False,
    ) -> np.ndarray:
        """
        Decodes a single sample of detection events.

        Parameters
        ----------
        det_events : 1D numpy array of int or bool, or list of int/bool
            The detection events for a single sample.
        threads : int, optional
            Number of decoder threads to use.
        verbose : bool, default False
            If True, streams Tesseract's verbose output to console in real-time.
        print_stats : bool, default False
            If True, prints out the number of shots (and number of errors, if known)
            during decoding.

        Returns
        -------
        prediction : 1D numpy array of bool
            The predicted observable flips.

        Raises
        ------
        RuntimeError
            If Tesseract decoding fails.
        ValueError
            If input array is not 1D or has an unsupported dtype after conversion,
            or if elements are not convertible to int/bool.
        TypeError
            If det_events cannot be converted to a NumPy array.
        """
        try:
            # Convert input to a NumPy array.
            det_events_np = np.array(det_events)
        except Exception as e:
            raise TypeError(f"det_events could not be converted to a NumPy array: {e}")

        if det_events_np.ndim != 1:
            raise ValueError(
                f"det_events must be a 1D array-like, got {det_events_np.ndim}D after conversion."
            )
        if not (
            det_events_np.dtype == bool
            or np.issubdtype(det_events_np.dtype, np.integer)
        ):
            raise ValueError(
                f"Elements of det_events must be boolean or integer-like. Detected dtype after conversion: {det_events_np.dtype}"
            )

        det_events_str = convert_array_to_01_format(det_events_np.reshape(1, -1))
        results = self._call_core_decoder(
            det_events_str,
            threads=threads,
            verbose_flag=verbose,
            print_stats_flag=print_stats,
        )

        if results["returncode"] != 0 or results["stderr"]:
            error_message = (
                results["stderr"].strip()
                if results["stderr"]
                else f"Tesseract decoding failed with return code {results['returncode']}."
            )
            raise RuntimeError(f"Tesseract decoding failed: {error_message}")

        outputs = results["output"]
        prediction = convert_01_format_to_array(outputs).astype(bool).ravel()

        return prediction

    def decode_batch(
        self,
        det_events: np.ndarray | List[List[int | bool]],
        *,
        threads: int | None = None,
        verbose: bool = False,
        print_stats: bool = False,
    ) -> np.ndarray:
        """
        Decodes a batch of detection event samples.

        Parameters
        ----------
        det_events : 2D numpy array of int or bool, or list of lists of int/bool
            The detection events for multiple samples. Each row is a sample.
        threads : int, optional
            Number of decoder threads to use.
        verbose : bool, default False
            If True, streams Tesseract's verbose output to console in real-time.
        print_stats : bool, default False
            If True, prints out the number of shots (and number of errors, if known)
            during decoding.

        Returns
        -------
        prediction : 2D numpy array of bool
            The predicted observable flips for each sample. Each row is a prediction.

        Raises
        ------
        RuntimeError
            If Tesseract decoding fails.
        ValueError
            If input array is not 2D or has an unsupported dtype after conversion,
            or if elements are not convertible to int/bool.
        TypeError
            If det_events cannot be converted to a NumPy array.
        """
        try:
            # Convert input to a NumPy array.
            det_events_np = np.array(det_events)
        except Exception as e:
            raise TypeError(f"det_events could not be converted to a NumPy array: {e}")

        if det_events_np.ndim != 2:
            raise ValueError(
                f"det_events must be a 2D array-like, got {det_events_np.ndim}D after conversion."
            )

        if (
            det_events_np.shape[0] == 0
        ):  # Handle empty batch early, after potential conversion
            num_observables = self.dem.num_observables if self.dem else 0
            return np.empty((0, num_observables), dtype=bool)

        if not (
            det_events_np.dtype == bool
            or np.issubdtype(det_events_np.dtype, np.integer)
        ):
            raise ValueError(
                f"Elements of det_events must be boolean or integer-like. Detected dtype after conversion: {det_events_np.dtype}"
            )

        det_events_str = convert_array_to_01_format(det_events_np)
        results = self._call_core_decoder(
            det_events_str,
            threads=threads,
            verbose_flag=verbose,
            print_stats_flag=print_stats,
        )

        if results["returncode"] != 0 or results["stderr"]:
            error_message = (
                results["stderr"].strip()
                if results["stderr"]
                else f"Tesseract decoding failed with return code {results['returncode']}."
            )
            raise RuntimeError(f"Tesseract decoding failed: {error_message}")

        outputs = results["output"]
        prediction = convert_01_format_to_array(outputs).astype(bool)

        return prediction

    def simulate(
        self,
        shots: int,
        *,
        sample_seed: int | None = None,
        threads: int | None = None,
        verbose: bool = False,
        print_stats: bool = False,
    ) -> np.ndarray:
        """
        Simulates shots from the circuit and decodes them to predict observable flips.

        This method uses the `core.decode_from_circuit_file` function to perform
        shot sampling from the `stim.Circuit` provided during decoder initialization
        and then decodes these shots.

        Parameters
        ----------
        shots : int
            The target number of shots to sample from the circuit and decode.
        sample_seed : int, optional
            Seed used when initializing the random number generator for sampling shots.
        threads : int, optional
            Number of decoder threads to use.
        verbose : bool, default False
            If True, streams Tesseract's verbose output to console in real-time.
        print_stats : bool, default False
            If True, prints out Tesseract's statistics (e.g., number of shots, errors)
            during simulation and decoding.

        Returns
        -------
        predictions : 2D numpy array of bool
            A 2D NumPy array where each row corresponds to a shot and columns represent
            the predicted observable flips. The shape of the array is (shots, num_observables).

        Raises
        ------
        ValueError
            If the `TesseractDecoder` was not initialized with a `stim.Circuit`,
            or if provided `shots` is not positive.
        RuntimeError
            If the Tesseract simulation or decoding process fails.
        FileNotFoundError
            If the Tesseract executable cannot be found.
        """
        if self.circuit is None:
            raise ValueError(
                "TesseractDecoder must be initialized with a stim.Circuit to use the simulate method."
            )

        if shots <= 0:
            raise ValueError("Parameter 'shots' must be positive.")

        # if max_errors is not None and max_errors <= 0:
        #     raise ValueError("Parameter 'max_errors' must be positive if provided.")

        temp_circuit_file_path = None
        temp_prediction_output_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".stim"
            ) as circ_f:
                self.circuit.to_file(circ_f.name)
                temp_circuit_file_path = circ_f.name

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".01"
            ) as pred_out_f:
                temp_prediction_output_file_path = pred_out_f.name

            core_specific_kwargs = {
                k: v
                for k, v in self.decoding_prms.items()
                if v is not None or isinstance(v, bool)
            }

            stream_output_for_core = verbose or print_stats

            results = decode_from_circuit_file(
                circuit_file=temp_circuit_file_path,
                sample_num_shots=shots,
                # max_errors=max_errors,
                sample_seed=sample_seed,
                out_file=temp_prediction_output_file_path,
                out_format="01",
                threads=threads,
                verbose=verbose,
                print_stats=print_stats,
                stream_output=stream_output_for_core,
                **core_specific_kwargs,
            )

            if results["returncode"] != 0:
                error_message = (
                    results["stderr"].strip()
                    if results["stderr"]
                    else f"Tesseract simulation/decoding failed with return code {results['returncode']}."
                )
                if not results["stderr"] and results["stdout"]:
                    error_message += (
                        f"\nSTDOUT from Tesseract:\n{results['stdout'].strip()}"
                    )
                raise RuntimeError(
                    f"Tesseract simulation/decoding failed: {error_message}"
                )

            output_str = ""
            if temp_prediction_output_file_path and os.path.exists(
                temp_prediction_output_file_path
            ):
                with open(temp_prediction_output_file_path, "r") as f_out:
                    output_str = f_out.read()
            else:
                raise RuntimeError(
                    "Tesseract reported success, but the prediction output file was not found or is empty."
                )

            predictions = convert_01_format_to_array(output_str).astype(bool)

            # The number of predictions is determined by Tesseract based on input conditions.
            # No strict check against 'shots' if max_errors could have caused early termination
            # or if 'shots' was not provided.

            return predictions

        finally:
            if temp_circuit_file_path and os.path.exists(temp_circuit_file_path):
                os.remove(temp_circuit_file_path)
            if temp_prediction_output_file_path and os.path.exists(
                temp_prediction_output_file_path
            ):
                os.remove(temp_prediction_output_file_path)
