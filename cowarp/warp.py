# ======================================================================
#  Correlation Optimized Warping (COW) – Main Interface
# ----------------------------------------------------------------------
# This module provides the primary public entry point for the
# Correlation Optimized Warping (COW) algorithm.
#
# It exposes a unified interface to multiple COW variants:
#
#   • Standard COW with manual segmentation
#       (cow_dynamic_no_edges)
#
#   • Automatic Segmentation COW (ASCOW)
#       (cow_auto_no_edges)
#
# In both implementations, the sample signal endpoints remain fixed
# (i.e., the first and last points are not subject to warping).
#
# ----------------------------------------------------------------------
# Author  : Guram Chaganava
# Created : 08.11.2025
# ======================================================================

from .cow_dynamic_no_edges import cow_dynamic_no_edges
from .cow_auto_no_edges import cow_auto_no_edges, \
    STATIONARY_POINTS, GAUSSIAN_FILTER


def warp(reference,
         sample,
         auto_segment=False,
         num_intervals=None,
         segment_length=None,
         slack=None,
         segmentation_type=STATIONARY_POINTS,
         deformation_coeff=None,
         filter_func_code=GAUSSIAN_FILTER,
         filter_func=None,
         filter_func_params=3,
         process_filtered_signals=False,
         min_interval_length=None,
         return_details=False,
         verbose=False
         ):
    """
    Perform Correlation Optimized Warping (COW) alignment between a
    reference signal and a sample signal.

    This function serves as the main public interface to the COW
    implementation. It dispatches to either:

        • `cow_dynamic_no_edges`  (manual segmentation)
        • `cow_auto_no_edges`     (automatic segmentation)

    depending on the value of `auto_segment`.

    Parameters
    ----------
    reference : array-like
        1D numeric array representing the reference signal.
    sample : array-like
        1D numeric array representing the signal to be warped to the reference.

    auto_segment : bool, default=False
        If False, uses standart COW with manual segmentation (`cow_dynamic_no_edges`).
        If True, automatically determines segment boundaries
        (`cow_auto_no_edges`).

    # --- standart COW parameters (auto_segment=False) ---

    num_intervals : int, optional
        Number of warping intervals. Mutually exclusive with `interval_length`.

    segment_length : int, optional
        Length of each interval. Mutually exclusive with `num_intervals`.
        If both are provided, `num_intervals` is redefined from `segment_length`.

    slack : int, optional
        Maximum allowed shift (±) for interval endpoints.
        Automatically determined if not provided.

    # --- Automatic segmentation parameters (auto_segment=True) ---

    segmentation_type : str or Enum, optional
        Strategy for determining reference boundaries. Can be one of the following:
            - 'stationary_points'
            - 'inflection_points'
            - 'peak_boundaries'

    deformation_coeff : float, optional
        Boundary deformation coefficient in [0, 1].
        Controls allowable boundary movement.

    filter_func_code : str, optional
        Identifier of predefined filtering method. Can be one of the following:
        - 'moving_average'
        - 'gaussian'
        - 'no_filter'

    filter_func : callable, optional
        Custom filtering function. If provided, overrides `filter_func_code`.

    filter_func_params : int or float, optional
        Parameter passed to the selected filter function
        (e.g., window size or sigma).

    process_filtered_signals : bool, default=False
        If True, segmentation and warping are performed on filtered signals.
        Otherwise, filtering is used only to determine boundaries.

    min_interval_length : int, optional
        Minimum allowed interval size. If omitted,
        a recommended default is automatically derived.

    # --- Output control ---

    return_details : bool, default=False
        If True, returns a dictionary containing additional data:
            - "warped_sample"
            - "correlation"
            - "warping_path"
            - "boundaries"
        Otherwise returns `(warped_sample, final_correlation)`.

    verbose : bool, default=False
        If True, prints diagnostic messages and intermediate results.

    Returns
    -------
    tuple or dict
        If `return_details=False`:
            warped_sample : np.ndarray
                Warped version of `sample`, aligned to the same
                length as `reference`.
            final_correlation : float
                Pearson correlation coefficient between
                `reference` and `warped_sample`.

        If `return_details=True`:
            dict with keys:
                - "warped_sample"
                - "correlation"
                - "warping_path"
                    List of boundary indices in the sample signal
                    corresponding to the fixed reference intervals.
                    Length equals number_of_intervals + 1.
                - "boundaries"
                    Fixed interval boundaries in the reference signal.
                    Length equals number_of_intervals + 1.

    Raises
    ------
    ValueError
        If input lengths are incompatible, signals are too short to warp,
        or parameters are inconsistent.

    TypeError
        If inputs cannot be interpreted as numeric 1D arrays.

    Notes
    -----
    * This function is intentionally lightweight and acts as a stable
      public API entry point.
    * Internal implementations may evolve without changing this interface.
    * All returned signals are guaranteed to be 1D NumPy float64 arrays.
    * Endpoints of the sample signal remain fixed (not warped).
    """
    if auto_segment:
        return cow_auto_no_edges(reference,
                                 sample,
                                 segmentation_type=segmentation_type,
                                 deformation_coeff=deformation_coeff,
                                 filter_func_code=filter_func_code,
                                 filter_func=filter_func,
                                 filter_func_params=filter_func_params,
                                 process_filtered_signals=process_filtered_signals,
                                 min_interval_length=min_interval_length,
                                 return_details=return_details,
                                 verbose=verbose)
    else:
        return cow_dynamic_no_edges(reference,
                                    sample,
                                    num_intervals=num_intervals,
                                    segment_length=segment_length,
                                    slack=slack,
                                    min_interval_length=min_interval_length,
                                    return_details=return_details,
                                    verbose=verbose)
