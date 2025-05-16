import os
import subprocess
from typing import Any, Dict, List, Optional


def _find_tesseract_executable() -> str:
    """
    Finds the tesseract executable.

    Returns
    -------
    str
        The path to the tesseract executable.

    Raises
    ------
    FileNotFoundError
        If the tesseract executable cannot be found.
    """
    current_dir = os.path.dirname(__file__)

    # 1. Check for prebuilt executable copied into the package
    path_prebuilt = os.path.join(
        current_dir,
        "_prebuilt_executable",
        "tesseract",
    )
    if os.path.exists(path_prebuilt) and os.access(path_prebuilt, os.X_OK):
        return os.path.abspath(path_prebuilt)

    # 2. Check original submodule path
    path_in_submodule = os.path.join(
        current_dir,
        "tesseract-decoder",
        "bazel-bin",
        "src",
        "tesseract",
    )
    if os.path.exists(path_in_submodule) and os.access(path_in_submodule, os.X_OK):
        return os.path.abspath(path_in_submodule)

    # 3. Check if the executable is in the system PATH
    import shutil

    path_in_system = shutil.which("tesseract")
    if path_in_system:
        return path_in_system

    raise FileNotFoundError(
        "Tesseract executable not found. "
        "Please ensure it is built and in your PATH, or copied to the "
        "'_prebuilt_executable' directory within the 'src/tesseract_decoder' directory "
        "before installation, as per README instructions. "
        f"Checked locations: '{path_prebuilt}', '{path_in_submodule}', and system PATH."
    )


TESSERACT_PATH = _find_tesseract_executable()


def run_tesseract_command(args: List[str]) -> Dict[str, Any]:
    """
    Runs a generic Tesseract command with the given arguments.

    Parameters
    ----------
    args : list of str
        A list of command-line arguments to pass to the Tesseract executable.

    Returns
    -------
    dict
        A dictionary containing the 'stdout', 'stderr', and 'returncode' of the command.

    Raises
    ------
    FileNotFoundError
        If the Tesseract executable is not found.
    subprocess.CalledProcessError
        If the Tesseract command returns a non-zero exit code, this error
        will be raised if check=True is used (not used here to allow manual error handling).
    """
    command = [TESSERACT_PATH] + args
    try:
        # For commands that produce a lot of output, or run for a long time,
        # consider using subprocess.Popen for more control.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,  # Decodes stdout and stderr as text
            check=False,  # Do not raise an exception for non-zero exit codes automatically
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The Tesseract executable was not found at '{TESSERACT_PATH}'. "
            "Please ensure it's correctly built and the path is configured."
        )
    except Exception as e:
        # Catch other potential subprocess errors
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,  # Indicate a wrapper/subprocess level error
        }


def decode_from_circuit_file(
    circuit_file: str,
    *,
    # Shot Sampling & Core Decoding Config
    sample_num_shots: Optional[int] = None,
    max_errors: Optional[int] = None,
    shot_range_begin: Optional[int] = None,
    shot_range_end: Optional[int] = None,
    sample_seed: Optional[int] = None,
    # Output Configuration
    out_file: Optional[str] = None,
    out_format: Optional[str] = None,
    dem_out_file: Optional[str] = None,
    stats_out_file: Optional[str] = None,
    # Decoding Algorithm & Performance
    threads: Optional[int] = None,
    beam: Optional[int] = None,
    pq_limit: Optional[int] = None,
    num_det_orders: Optional[int] = None,
    det_order_seed: Optional[int] = None,
    det_penalty: Optional[float] = None,
    # Heuristics
    at_most_two_errors_per_detector: bool = False,
    beam_climbing: bool = False,
    no_revisit_dets: bool = False,
    no_merge_errors: bool = False,
    # Verbosity/Stats
    print_stats: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Decodes error events by sampling shots from a Stim circuit file.
    Corresponds to: ./tesseract --circuit CIRCUIT_FILE.stim --sample-num-shots N [...].

    Either `sample_num_shots` or `max_errors` must be provided.

    Examples
    --------
    >>> # Sample 1000 shots from a circuit and save predictions
    >>> results = decode_from_circuit_file(
    ...     "surface_code.stim",
    ...     sample_num_shots=1000,
    ...     out_file="predictions.01",
    ... )
    >>> print(f"Stdout: {results['stdout']}")

    >>> # Example with advanced options
    >>> advanced_results = decode_from_circuit_file(
    ...     "circuit_file.stim",
    ...     sample_num_shots=10000,
    ...     pq_limit=1000000,
    ...     at_most_two_errors_per_detector=True,
    ...     det_order_seed=232852747,
    ...     sample_seed=232856747,
    ...     threads=32,
    ...     print_stats=True,
    ...     beam=23,
    ...     num_det_orders=1,
    ...     shot_range_begin=582,
    ...     shot_range_end=583
    ... )
    >>> print(f"Advanced Stdout: {advanced_results['stdout']}")

    Parameters
    ----------
    circuit_file : str
        Stim circuit file path
    sample_num_shots : int, optional
        If provided, will sample the requested number of shots from the Stim circuit and
        decode them. May end early if `max-errors` errors are reached before decoding all
        shots.
    max_errors : int, optional
        If provided, will sample at least this many errors from the Stim circuit and
        decode them.
    shot_range_begin : int, optional
        Useful for processing a fragment of a file. If shot_range_begin == 0 and shot_range_end == 0 (the default), then all available shots will be decoded. Otherwise, only those in the range [shot_range_begin, shot_range_end) will be decoded.
    shot_range_end : int, optional
        Useful for processing a fragment of a file. If shot_range_begin == 0 and shot_range_end == 0 (the default), then all available shots will be decoded. Otherwise, only those in the range [shot_range_begin, shot_range_end) will be decoded.
    sample_seed : int, optional
        Seed used when initializing the random number generator for sampling shots
    out_file : str, optional
        File to write observable flip predictions to (stdout if None)
    out_format : str, optional
        Format of the file to write observable flip predictions to (01/b8/dets/hits/ptb64/r8).
        If not provided, the format will be inferred from the `out_file` extension or
        will be set to "01" if `out_file=None` or the extension is not included in the
        supported formats.
    dem_out_file : str, optional
        File to write matching frequency dem to
    stats_out_file : str, optional
        File to write high-level statistics and metadata to
    threads : int, optional
        Number of decoder threads to use
    beam : int, optional
        Beam to use for truncation (default = infinity)
    pq_limit : int, optional
        Maximum size of the priority queue (default = infinity)
    num_det_orders : int, optional
        Number of ways to orient the manifold when reordering the detectors
    det_order_seed : int, optional
        Seed used when initializing the random detector traversal orderings
    det_penalty : float, optional
        Penalty cost to add per activated detector in the residual syndrome.
    at_most_two_errors_per_detector : bool, default False
        Use heuristic limitation of at most 2 errors per detector
    beam_climbing : bool, default False
        Use beam-climbing heuristic
    no_revisit_dets : bool, default False
        Use no-revisit-dets heuristic
    no_merge_errors : bool, default False
        If provided, will not merge identical error mechanisms.
    print_stats : bool, default False
        Prints out the number of shots (and number of errors, if known) during decoding.
    verbose : bool, default False
        Increases output verbosity

    Returns
    -------
    dict
        A dictionary containing the 'stdout', 'stderr', and 'returncode' of the Tesseract command.
        If `print_stats` is True, stdout will contain the statistics.
        If `out_file` is specified, predictions are written to that file.
    """
    supported_file_formats = {"01", "b8", "dets", "hits", "ptb64", "r8"}
    if out_file is None:
        out_file = "-"
        if out_format is None:
            out_format = "01"

    elif out_format is None:
        out_format = out_file.split(".")[-1]
        if out_format not in supported_file_formats:
            out_format = "01"

    args = ["--circuit", circuit_file, "--sample-num-shots", str(sample_num_shots)]

    if print_stats:
        args.append("--print-stats")
    if pq_limit is not None:
        args.extend(["--pqlimit", str(pq_limit)])
    if at_most_two_errors_per_detector:
        args.append("--at-most-two-errors-per-detector")
    if det_order_seed is not None:
        args.extend(["--det-order-seed", str(det_order_seed)])
    if sample_seed is not None:
        args.extend(["--sample-seed", str(sample_seed)])
    if threads is not None:
        args.extend(["--threads", str(threads)])
    if beam is not None:
        args.extend(["--beam", str(beam)])
    if num_det_orders is not None:
        args.extend(["--num-det-orders", str(num_det_orders)])
    if shot_range_begin is not None:
        args.extend(["--shot-range-begin", str(shot_range_begin)])
    if shot_range_end is not None:
        args.extend(["--shot-range-end", str(shot_range_end)])
    if out_file is not None:
        args.extend(["--out", out_file])
        if out_format is not None:
            args.extend(["--out-format", out_format])
    if dem_out_file is not None:
        args.extend(["--dem-out", dem_out_file])

    if no_merge_errors:
        args.append("--no-merge-errors")
    if max_errors is not None:
        args.extend(["--max-errors", str(max_errors)])
    if stats_out_file is not None:
        args.extend(["--stats-out", stats_out_file])
    if verbose:
        args.append("--verbose")

    if det_penalty is not None:
        args.extend(["--det-penalty", str(det_penalty)])
    if beam_climbing:
        args.append("--beam-climbing")
    if no_revisit_dets:
        args.append("--no-revisit-dets")

    return run_tesseract_command(args)


def decode_from_detection_events(
    dem_file: str,
    detection_event_file: str,
    obs_in_file: Optional[str] = None,
    *,
    # Input Format & Configuration
    detection_event_format: Optional[str] = None,
    obs_in_format: Optional[str] = None,
    in_includes_appended_observables: bool = False,
    # Output Configuration
    out_file: Optional[str] = None,
    out_format: Optional[str] = None,
    stats_out_file: Optional[str] = None,
    # Decoding Algorithm & Performance
    threads: Optional[int] = None,
    beam: Optional[int] = None,
    pq_limit: Optional[int] = None,
    num_det_orders: Optional[int] = None,
    det_order_seed: Optional[int] = None,
    det_penalty: Optional[float] = None,
    # Heuristics
    at_most_two_errors_per_detector: bool = False,
    beam_climbing: bool = False,
    no_revisit_dets: bool = False,
    no_merge_errors: bool = False,
    # Verbosity/Stats
    print_stats: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Decodes errors from a pre-existing detection event file using a DEM file.
    Corresponds to: ./tesseract --in EVENTS_FILE --in-format FORMAT --dem DEM_FILE --out DECODED_FILE [...]

    Examples
    --------
    >>> # Decode from a detection event file using a DEM
    >>> results_no_obs = decode_from_detection_events(
    ...     "surface_code.dem",
    ...     "events.01",
    ...     out_file="decoded_no_obs.txt"
    ... )
    >>> print(f"Stdout: {results_no_obs['stdout']}")

    >>> # Decode from a detection event file and observable flips file using a DEM
    >>> results_with_obs = decode_from_detection_events(
    ...     "surface_code.dem",
    ...     "events.01",
    ...     "obs.01",
    ...     out_file="decoded_with_obs.txt"
    ... )
    >>> print(f"Stdout: {results_with_obs['stdout']}")

    Parameters
    ----------
    dem_file : str
        Stim dem file path
    detection_event_file : str
        File to read detection events (and possibly observable flips) from
    obs_in_file : str, optional
        File to read observable flips from
    detection_event_format : str, optional
        Format of the file to read detection events from (01/b8/dets/hits/ptb64/r8).
        If not provided, the format will be inferred from the `detection_event_file`
        extension.
    obs_in_format : str, optional
        Format of the file to observable flips from (01/b8/dets/hits/ptb64/r8).
        If not provided, the format will be inferred from the `obs_in_file` extension.
    in_includes_appended_observables : bool, default False
        If present, assumes that the observable flips are appended to the end of each shot.
    out_file : str, optional
        File to write observable flip predictions to (stdout if None)
    out_format : str, optional
        Format of the file to write observable flip predictions to (01/b8/dets/hits/ptb64/r8).
        If not provided, the format will be inferred from the `out_file` extension or
        will be set to "01" if `out_file=None` or the extension is not included in the
        supported formats.
    stats_out_file : str, optional
        File to write high-level statistics and metadata to
    threads : int, optional
        Number of decoder threads to use
    beam : int, optional
        Beam to use for truncation (default = infinity)
    pq_limit : int, optional
        Maximum size of the priority queue (default = infinity)
    num_det_orders : int, optional
        Number of ways to orient the manifold when reordering the detectors
    det_order_seed : int, optional
        Seed used when initializing the random detector traversal orderings.
    det_penalty : float, optional
        Penalty cost to add per activated detector in the residual syndrome.
    at_most_two_errors_per_detector : bool, default False
        Use heuristic limitation of at most 2 errors per detector
    beam_climbing : bool, default False
        Use beam-climbing heuristic
    no_revisit_dets : bool, default False
        Use no-revisit-dets heuristic
    no_merge_errors : bool, default False
        If provided, will not merge identical error mechanisms.
    print_stats : bool, default False
        Prints out the number of shots (and number of errors, if known) during decoding.
    verbose : bool, default False
        Increases output verbosity

    Returns
    -------
    dict
        A dictionary containing the 'stdout', 'stderr', and 'returncode' of the Tesseract command.
    """
    supported_file_formats = {"01", "b8", "dets", "hits", "ptb64", "r8"}

    if out_file is None:
        out_file = "-"
        if out_format is None:
            out_format = "01"

    elif out_format is None:
        out_format = out_file.split(".")[-1]
        if out_format not in supported_file_formats:
            out_format = "01"

    if detection_event_format is None:
        detection_event_format = detection_event_file.split(".")[-1]
    if obs_in_file is not None and obs_in_format is None:
        obs_in_format = obs_in_file.split(".")[-1]

    args = [
        "--dem",
        dem_file,
        "--in",
        detection_event_file,
        "--in-format",
        detection_event_format,
    ]

    if obs_in_file is not None:
        args.extend(["--obs_in", obs_in_file])
        if obs_in_format is not None:
            args.extend(["--obs-in-format", obs_in_format])

    if out_file is not None:
        args.extend(["--out", out_file])
        if out_format is not None:
            args.extend(["--out-format", out_format])

    if print_stats:
        args.append("--print-stats")
    if pq_limit is not None:
        args.extend(["--pqlimit", str(pq_limit)])
    if at_most_two_errors_per_detector:
        args.append("--at-most-two-errors-per-detector")
    if det_order_seed is not None:
        args.extend(["--det-order-seed", str(det_order_seed)])
    if threads is not None:
        args.extend(["--threads", str(threads)])
    if beam is not None:
        args.extend(["--beam", str(beam)])
    if num_det_orders is not None:
        args.extend(["--num-det-orders", str(num_det_orders)])

    if in_includes_appended_observables:
        args.append("--in-includes-appended-observables")
    if no_merge_errors:
        args.append("--no-merge-errors")
    if stats_out_file is not None:
        args.extend(["--stats-out", stats_out_file])
    if verbose:
        args.append("--verbose")

    if det_penalty is not None:
        args.extend(["--det-penalty", str(det_penalty)])
    if beam_climbing:
        args.append("--beam-climbing")
    if no_revisit_dets:
        args.append("--no-revisit-dets")

    return run_tesseract_command(args)
