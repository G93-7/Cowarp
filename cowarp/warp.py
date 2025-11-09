
# ======================================================================
#  Correlation Optimized Warping (COW) – Main Interface
# ----------------------------------------------------------------------
# This module provides the main entry point for the COW algorithm.
# It serves as a unified interface to different COW variants,
# including the 'dynamic no-edges' implementation and future extensions.

# This is the initial release of the package.
# It currently includes a single implementation of the COW algorithm,
# in which the sample signal endpoints remain fixed (i.e., not subject to warping).
#
#
#  Author: [Guram Chaganava]
#  Created: [08.11.2025]
# ======================================================================

from cow_dynamic_no_edges_final import cow_dynamic_no_edges


def warp(reference,
         sample,
         num_intervals=None,
         interval_length=None,
         slack=None,
         min_interval_length=None,
         return_details=False,
         verbose=False
         ):
    """
    Perform Correlation Optimized Warping (COW) alignment between a reference
    signal and a sample signal using the default COW implementation
    (`cow_dynamic_no_edges`).

    Parameters
    ----------
    reference : array-like
        1D numeric array representing the reference signal.
    sample : array-like
        1D numeric array representing the signal to be warped to the reference.
    num_intervals : int, optional
        Number of warping intervals. Mutually exclusive with `interval_length`.
    interval_length : int, optional
        Length of each interval. Used if `num_of_intervals`
        is not provided.
    slack : int, optional
        Maximum allowed shift (±) for interval endpoints. Automatically
        determined if not provided.
    min_interval_length : int, optional
        Minimal allowed interval size. If omitted,
        a recommended default is automatically derived.
    return_details : bool, default=False
        If True, returns a dictionary containing additional data:
        - "warped_sample"
        - "correlation"
        - "warping_path"
        - "boundaries"
        Otherwise returns `(warped_signal, final_correlation)`.
    verbose : bool, default=False
        If True, prints diagnostic messages and intermediate results.

    Returns
    -------
    tuple or dict
        If `return_details=False`:
            warped_sample : np.ndarray
                The warped version of `sample`, aligned to the same length as `reference`.
            final_correlation : float
                Pearson correlation coefficient between `reference` and `warped_sample`.
        If `return_details=True`:
            dict with keys:
                - "warped_sample"
                - "correlation"
                - "warping_path"
                    List of boundary indices in the sample signal defining the intervals
                    that correspond to the fixed reference intervals.
                    The length of this list is equal to num_intervals + 1.
                - "boundaries"
                    List of fixed interval boundaries in the reference signal.
                    The length of this list is equal to num_intervals + 1.
    Raises
    ------
    ValueError
        If input lengths are incompatible, signals are too short to warp or
        inputs contain inconsistent values.
    TypeError
        If inputs cannot be interpreted as numeric 1D arrays.
    MemoryError
        If internal conversions or resampling exceed system memory.

    Notes
    -----
    * This function is intentionally thin; it forwards arguments to the
      selected implementation (`cow_dynamic_no_edges`).
    * The goal is to keep `cow()` as a stable public API, even if internal
      implementations change or multiple algorithms are added later.
    * All returned signals are guaranteed to be 1D NumPy float64 arrays.

    Examples
    --------
    # >>> from test_pa.main import cow
    # >>> warped, corr = cow(ref_signal, sample_signal, num_intervals=10)
    # >>> corr
    0.9824
    """
    return cow_dynamic_no_edges(reference,
                                sample,
                                num_intervals=num_intervals,
                                interval_length=interval_length,
                                slack=slack,
                                min_interval_length=min_interval_length,
                                return_details=return_details,
                                verbose=verbose)
