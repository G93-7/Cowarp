
# Correlation Optimized Warping (COW) - dynamic programming implementation.

# The author refers to this version of COW as a *dynamic* because
# each interval’s possible start positions depend on the
# start positions of the preceding interval, resulting in a dynamic
# number of valid configurations.

# The term *no edges* means that the sample signal edges (first and last points)
# are not warped — they remain fixed during the alignment process.

# Works fully with NumPy arrays.
# Symmetric structure, increasing number of possible border positions from edge to center.
# Vectorized solution for speed-up.

import numpy as np


def validate_input_array(input_data):
    """
    Validate that the input is 1D numeric data and return it as a NumPy float64 array.

    Parameters
    ----------
    input_data : scalar, list, tuple, or array-like
        Input data representing a 1D numeric signal. Accepts scalars, Python sequences,
        NumPy arrays, or any object convertible to a NumPy array.

    Returns
    -------
    np.ndarray
        A 1D array of dtype float64. Scalars are converted to a length-1 array.

    Raises
    ------
    TypeError
        If input cannot be converted to a numeric array.
    ValueError
        If the resulting array is not 1D or contains non-numeric data.
    MemoryError
        If system runs out of memory during conversion.

    Notes
    -----
    * This function guarantees:
        - Output is always `np.float64`
        - Output is always 1D
    * Multidimensional data not supported yet
    """
    # Try converting to array first (handles scalars and sequences)
    try:
        numeric_array = np.asarray(input_data)
    except Exception as e:
        raise TypeError(f"[cowarp] input cannot be converted to array: {e}")

    # enforces scalar → 1D conversion and preserves shape otherwise
    arr = np.atleast_1d(numeric_array)

    # Must be 1D only
    arr_dim = numeric_array.ndim
    if arr_dim != 1:
        raise ValueError(f"[cowarp] input must be 1D array, got {arr_dim}D array")

    # Validate numeric dtype
    if not np.issubdtype(numeric_array.dtype, np.number):
        raise ValueError(f"[cowarp] input contains non-numeric data (dtype={arr.dtype})")

    # Any NaN present is invalid
    if np.isnan(numeric_array).any():
        raise ValueError("[cowarp] input contains NaN values — clean or interpolate before calling this function")

    # Convert to float64 if needed (copy=False avoids unnecessary copy)
    try:
        return numeric_array.astype(np.float64, copy=False)
    except MemoryError as e:
        raise MemoryError(f"[cowarp] insufficient memory to convert input to float64: {e}")


def validate_input(input):
    """
    Validate and convert a single numeric input to an integer.

    Accepts:
        • Python int or float
        • NumPy scalar (0-D array)
        • Raises an error if input is non-numeric, NaN, or not scalar

    Conversion rules:
        • Integer inputs are returned unchanged
        • Float inputs are rounded to the nearest integer
        • NumPy scalar values are converted to Python scalars first
        • Non-scalar NumPy arrays are rejected

    Parameters
    ----------
    input : int, float, numpy scalar, or array-like
        Input value expected to represent a single numeric quantity.

    Returns
    -------
    int
        Integer value after validation and (if needed) rounding.

    Raises
    ------
    ValueError
        If input is an array, non-numeric, or cannot be safely converted.
    """

    # Handle NumPy scalar arrays (0D only)
    if isinstance(input, np.ndarray):
        if input.ndim != 0:
            raise ValueError("[cowarp] Input must be a scalar, not an array.")
        input = input.item()  # Extract Python scalar

    # Integer type → return as-is
    if isinstance(input, (int, np.integer)):
        return int(input)

    # Float type → round to nearest integer
    if isinstance(input, (float, np.floating)):
        if np.isnan(input):
            raise ValueError("[cowarp] Input cannot be NaN.")
        return int(np.rint(input))

    # Anything else → reject
    raise ValueError("[cowarp] Input must be a numeric scalar (int or float).")


def has_valid_len(arr_len, min_len):
    """
    Check if a signal length meets the minimum length requirement.
    """
    return arr_len >= min_len


def is_warpable_case_no_edges(arr_len, min_len):
    """
    Check if a signal is warpable under COW rules when sample signal edges are fixed:
    A signal is warpable only if its length satisfies:
        length >= 2 * min_interval_length + 1
    """
    return arr_len >= 2 * min_len + 1


def determine_min_interval_length(min_interval_length, verbose=False):
    """
    Ensure `min_interval_length` is a valid integer.
    If None is provided, replaces it with a default value.

    Parameters
    ----------
    min_interval_length : int or float or None
        User-provided minimum interval length. If None, default is used.

    verbose : bool, optional
        If True, prints information messages instead of being silent.

    Returns
    -------
    int
        Validated minimum interval length (>= 2).

    Raises
    ------
    ValueError
        If the resulting value is < 2 or the input is inconsistent
    """
    if min_interval_length is None:
        min_interval_length = 3
        if verbose:
            print(f"[cowarp] min_interval_length not provided, using default={min_interval_length}")

    min_interval_length = validate_input(min_interval_length)

    if min_interval_length < 2:
        raise ValueError(f"[cowarp] min_interval_length={min_interval_length} is invalid; must be >= 2")

    if verbose:
        print(f"[cowarp] Using min_interval_length = {min_interval_length}")

    return min_interval_length


def determine_num_intervals(
        num_intervals,
        interval_length,
        signal_length,
        min_interval_length,
        verbose=False,
):
    """
    Determine the number of warping intervals for Correlation Optimized Warping (COW).

    Parameters
    ----------
    num_intervals : int or None
        Number of intervals to use. If None, it will be computed automatically.
    interval_length : int or None
        Length of each interval. Used only if `num_intervals` is not provided.
    signal_length : int
        Total length of the signal to be warped.
    min_interval_length : int
        Minimum allowed size for each interval.
    verbose : bool, optional
        If True, prints internal decisions and computed values. Defaults to False.

    Returns
    -------
    int
        Validated number of intervals to use.

    Raises
    ------
    ValueError
        If computed or provided number of intervals is outside valid bounds
        or input parameters are inconsistent.

    Notes
    -----
    - If `num_intervals` is None and `interval_length` is provided,
      it computes `num_intervals = signal_length // interval_length`.
    - If neither is provided, defaults to half of the maximum valid interval count.
    - Ensures the result satisfies:
          2 ≤ num_intervals ≤ signal_length // min_interval_length
    """
    # Define allowable range
    min_allowed = 2
    max_allowed = signal_length // min_interval_length
    if signal_length % min_interval_length == 0:
        max_allowed -= 1  # avoid boundary issue

    # Auto-compute if not provided
    if num_intervals is None:
        if interval_length is not None:
            interval_length = validate_input(interval_length)
            num_intervals = signal_length // interval_length
            if verbose:
                print(f"[cowarp] num_intervals auto-set from interval_length → {num_intervals}")
        else:
            num_intervals = max(max_allowed // 2, 2)
            if verbose:
                print(f"[cowarp] num_intervals auto-set to → {num_intervals}")

    # Validate numeric input
    num_intervals = validate_input(num_intervals)

    # Check validity range
    if num_intervals < min_allowed:
        raise ValueError(
            f"[cowarp] num_intervals={num_intervals} is below the minimum allowed ({min_allowed})"
        )
    if num_intervals > max_allowed:
        raise ValueError(
            f"[cowarp] num_intervals={num_intervals} exceeds the maximum allowed ({max_allowed})"
        )

    if verbose:
        print(f"[cowarp] Using num_intervals = {num_intervals}")

    return num_intervals


def determine_slack(slack, interval_length, min_interval_length, verbose=False):
    """
    Validate and return a usable slack value.

    Parameters
    ----------
    slack : int, float, or None
        User-specified slack. If None, the half of the maximum allowed slack is used.
    interval_length : int
        Length of the interval being processed.
    min_interval_length : int
        Minimum allowed interval length, used to compute max slack.
    verbose : bool, optional
        If True, prints informational messages (default: False).

    Returns
    -------
    int
        Validated slack value.

    Raises
    ------
    ValueError
        If slack is below minimum or above maximum allowed
        or input parameters are inconsistent.
    """
    min_slack = 1
    max_slack = max(1, interval_length - min_interval_length)

    if slack is None:
        slack = max(max_slack // 2, 1)
        if verbose:
            print(f"[cowarp] Slack auto-set to -> {slack}")

    slack = validate_input(slack)

    # Check validity range
    if slack < min_slack:
        raise ValueError(
            f"[cowarp] slack={slack} is below the minimum allowed ({min_slack})"
        )
    if slack > max_slack:
        raise ValueError(
            f"[cowarp] num_intervals={slack} exceeds the maximum allowed ({max_slack})"
        )

    if verbose:
        print(f"[cowarp] Using slack = {slack}")

    return slack


def get_boundary_indices(num_intervals, interval_length, signal_length, verbose=False):
    """
    Generate boundary indices for splitting a signal into equal intervals.

    Parameters
    ----------
    num_intervals : int
        Number of intervals.
    interval_length : int
        Length of each interval.
    signal_length : int
        Total length of the signal.
    verbose : bool, optional
        If True, prints boundary indices (default: False).

    Returns
    -------
    list[int]
        List of boundary indices, ending with `signal_length`.
    """
    # Use list comprehension instead of loop + append → faster & cleaner
    boundaries = [i * interval_length for i in range(num_intervals)]
    boundaries.append(signal_length)
    if verbose:
        print(f"[cowarp] boundary indices -> {boundaries}")
    return boundaries


def interpolate_signal(signal, old_len, new_len):
    """
    Linearly resamples a 1D signal from old_len to new_len

    Parameters
    ----------
    signal : np.ndarray
        Input 1D numeric signal.
    old_len : int
        Original length of signal.
    new_len : int
        Desired resampled length.

    Returns
    -------
    np.ndarray
        Resampled signal of length new_len.
    """
    # Generate fractional index positions
    xi = np.linspace(0, old_len - 1, new_len)

    # Get integer left indices
    left = xi.astype(int)

    # Avoid out-of-bounds when left == old_len - 1
    np.minimum(left, old_len - 2, out=left)

    # Fractional distance for interpolation
    frac = xi - left

    # Linear interpolation: y = y0 + t * (y1 - y0)
    return signal[left] + frac * (signal[left + 1] - signal[left])


def calculate_correlation(signal1, signal2):
    """
    Compute the Pearson correlation coefficient between two 1D arrays.

    Parameters
    ----------
    signal1 : np.ndarray
        First 1D numeric array.
    signal2 : np.ndarray
        Second 1D numeric array.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].

    Notes
    -----
    - Returns 1.0 if both signals are constant (zero variance).
    - Returns 0.0 if one signal is constant and the other is not.
    """
    # # Ensure 1D numpy arrays
    # signal1 = np.asarray(signal1, dtype=np.float64).ravel()
    # signal2 = np.asarray(signal2, dtype=np.float64).ravel()

    # Center the signals
    a_c = signal1 - signal1.mean()
    b_c = signal2 - signal2.mean()

    # Compute norms
    norm_a = np.linalg.norm(a_c)
    norm_b = np.linalg.norm(b_c)

    # Handle constant signals
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Pearson correlation via dot product
    return np.dot(a_c, b_c) / (norm_a * norm_b)


def precompute_interpolation_params(segment_len, last_segment_len, min_interval_length, slack):
    """
    Precompute interpolation indices and coefficients for dynamic programming
    in Correlation Optimized Warping (COW). Only lengths actually used are computed:
    segment_len ± slack and
    last_segment_len ± slack if last_segment_len != segment_len

    Parameters
    ----------
    segment_len : int
        Standard segment length.
    last_segment_len : int
        Length of the last segment (may differ from standard).
    min_interval_length : int
        Minimum allowed interval length.
    slack : int
        Maximum allowed deviation from segment length.

    Returns
    -------
    interp_inds : np.ndarray
        Precomputed left indices for interpolation for segment_len ± slack.
    interp_coeffs : np.ndarray
        Precomputed interpolation coefficients for segment_len ± slack.
    interp_inds_last : np.ndarray or None
        Precomputed left indices for the last segment, if different.
    interp_coeffs_last : np.ndarray or None
        Precomputed interpolation coefficients for the last segment, if different.
    """

    def compute_interp_params(target_len):
        """
        Helper to precompute interpolation indices and coefficients
        for a given segment length ± slack.
        """
        min_len = max(min_interval_length, target_len - slack)
        max_len = target_len + slack
        num_lens = max_len - min_len + 1

        inds = np.empty((num_lens, target_len), dtype=int)
        coeffs = np.empty((num_lens, target_len), dtype=float)

        for i, old_len in enumerate(range(max_len, min_len - 1, -1)):
            xi = np.linspace(0, old_len - 1, target_len)
            left = xi.astype(int)
            np.minimum(left, old_len - 2, out=left)  # avoid out-of-bounds
            inds[i, :] = left
            coeffs[i, :] = xi - left

        return inds, coeffs

    # Precompute for standard segment
    interp_inds, interp_coeffs = compute_interp_params(segment_len)

    # Precompute for last segment if different
    if segment_len != last_segment_len:
        interp_inds_last, interp_coeffs_last = compute_interp_params(last_segment_len)
    else:
        interp_inds_last = None
        interp_coeffs_last = None

    return interp_inds, interp_coeffs, interp_inds_last, interp_coeffs_last


def fill_correlation_matrix(reference, sample, num_intervals, slack, interval_len,
                            min_interval_len, boundary_indices):
    """
    Fill the cumulative correlation matrix for Correlation Optimized Warping (COW).

    Parameters
    ----------
    reference : np.ndarray
        Reference signal (1D array)
    sample : np.ndarray
        Sample signal (1D array)
    num_intervals : int
        Number of intervals
    slack : int
        Maximum deviation from interval length
    interval_len : int
        Standard interval length
    min_interval_len : int
        Minimum allowed interval length
    boundary_indices : list[int]
        Indices defining segment boundaries in the reference

    Returns
    -------
    best_cumulative_correlations : np.ndarray
        Matrix of best cumulative correlations
    possible_start_borders : np.ndarray
        Matrix of possible start borders
    best_end_borders : np.ndarray
        Matrix of best end borders
    """

    # Adjust min/max interval lengths
    min_len = max(min_interval_len, interval_len - slack)
    max_len = interval_len + slack

    max_num_borders = 2 * slack * num_intervals // 2 + 1

    # Initialize DP matrices
    best_cumulative_correlations = np.full((num_intervals, max_num_borders), -1.0, dtype=float)
    possible_start_borders = np.full((num_intervals, max_num_borders), -1, dtype=int)
    best_end_borders = np.full((num_intervals - 1, max_num_borders), -1, dtype=int)

    # Precompute reference segments and norms
    ref_segments = []
    ref_norms = []
    for k in range(num_intervals):
        seg = reference[boundary_indices[k]:boundary_indices[k + 1]].astype(float)
        seg_cent = seg - seg.mean()
        ref_segments.append(seg_cent)
        ref_norms.append(np.linalg.norm(seg_cent))

    # Precompute interpolation parameters for last segment
    last_segment_len = len(ref_segments[-1])
    interp_inds, interp_coeffs, interp_inds_last, interp_coeffs_last = \
        precompute_interpolation_params(interval_len, last_segment_len, min_interval_len, slack)

    # Helper function: compute correlation with zero-norm handling
    def compute_corr(a, b, norm_a, norm_b):
        if norm_a == 0 and norm_b == 0:
            return 1.0
        elif norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    # Last segment
    start_border_idx = 0
    interval_start = boundary_indices[-2] - slack
    interval_end = boundary_indices[-2] + slack + 1
    ref_cent = ref_segments[-1]
    ref_norm = ref_norms[-1]

    for new_start in range(interval_start, interval_end):
        new_len = boundary_indices[-1] - new_start
        if new_len < min_interval_len:
            continue

        sample_seg = sample[new_start:boundary_indices[-1]]

        # Interpolation if needed
        if len(sample_seg) != last_segment_len:
            len_index = new_start - interval_start
            ind = interp_inds_last[len_index] if last_segment_len != interval_len else interp_inds[len_index]
            coeff = interp_coeffs_last[len_index] if last_segment_len != interval_len else interp_coeffs[len_index]
            sample_seg = sample_seg[ind] + coeff * (sample_seg[ind + 1] - sample_seg[ind])

        sample_cent = sample_seg - sample_seg.mean()
        sample_norm = np.linalg.norm(sample_cent)
        best_cumulative_correlations[0, start_border_idx] = compute_corr(ref_cent, sample_cent, ref_norm, sample_norm)
        possible_start_borders[0, start_border_idx] = new_start
        start_border_idx += 1

    # Inner segments and first segment
    end_border_first = interval_start
    end_border_last = interval_end

    for interval_idx in range(1, num_intervals):
        ref_cent = ref_segments[num_intervals - interval_idx - 1]
        ref_norm = ref_norms[num_intervals - interval_idx - 1]

        slack_mult = min(interval_idx + 1, num_intervals - interval_idx - 1)
        max_border_shift = slack * slack_mult

        start_border_idx = 0
        interval_start = boundary_indices[num_intervals - interval_idx - 1] - max_border_shift
        interval_end = boundary_indices[num_intervals - interval_idx - 1] + max_border_shift + 1

        first_valid = interval_start
        last_valid = interval_end

        for new_start in range(interval_start, interval_end):
            # candidate interval lengths
            interval_lengths = np.arange(min_len, max_len + 1)
            new_end_indices = new_start + interval_lengths
            mask = (new_end_indices >= end_border_first) & (new_end_indices < end_border_last)
            if not np.any(mask):
                first_valid += 1 if new_start == interval_start else 0
                continue

            last_valid = new_start

            valid_lengths = interval_lengths[mask]
            valid_end_indices = new_end_indices[mask]

            # Build segment matrix
            max_seg_len = max(valid_lengths)
            num_of_segments = len(valid_lengths)
            seg_mat = np.full((len(valid_lengths), max_seg_len), np.nan, dtype=float)
            for i, l in enumerate(valid_lengths):
                seg_mat[i, :l] = sample[new_start:valid_end_indices[i]]

            # Interpolate segments
            valid_len_idx = max_len - valid_lengths
            inds = interp_inds[valid_len_idx]
            coeffs = interp_coeffs[valid_len_idx]

            row_idx = np.arange(num_of_segments)[:, None]
            seg_interp = seg_mat[row_idx, inds]
            seg_interp_next = seg_mat[row_idx, inds + 1]
            seg_interp += coeffs * (seg_interp_next - seg_interp)

            # Center and compute norms
            seg_cent = seg_interp - seg_interp.mean(axis=1, keepdims=True)
            seg_norms = np.linalg.norm(seg_cent, axis=1)
            dots = seg_cent @ ref_cent
            den = ref_norm * seg_norms

            corrs = np.zeros_like(seg_norms, dtype=float)
            # both norms are zero case
            both_zero_mask = (seg_norms == 0) & (ref_norm == 0)
            corrs[both_zero_mask] = 1.0
            # normal case
            valid_mask = den != 0
            corrs[valid_mask] = dots[valid_mask] / den[valid_mask]

            # Add cumulative correlations
            corrs += best_cumulative_correlations[interval_idx - 1, valid_end_indices - end_border_first]

            best_idx = np.argmax(corrs)
            best_val = corrs[best_idx]
            best_end = valid_end_indices[best_idx]

            possible_start_borders[interval_idx, start_border_idx] = new_start
            best_end_borders[interval_idx - 1, start_border_idx] = best_end
            best_cumulative_correlations[interval_idx, start_border_idx] = best_val

            start_border_idx += 1

        end_border_first = first_valid
        end_border_last = last_valid + 1

    return best_cumulative_correlations, possible_start_borders, best_end_borders


def calculate_optimal_warping_path(best_cumulative_correlations,
                                   possible_start_borders,
                                   best_end_borders,
                                   number_of_intervals,
                                   boundary_indices,
                                   verbose=False):
    """
    Reconstruct the optimal warping path by backtracking through the
    dynamic programming matrices.

    Parameters
    ----------
    best_cumulative_correlations : ndarray, shape (num_intervals, num_states)
        Matrix of maximum cumulative correlations at each interval and state.

    possible_start_borders : ndarray, shape (num_intervals, num_states)
        For each interval and state, stores the valid starting index of the sample segment.

    best_end_borders : ndarray, shape (num_intervals - 1, num_states)
        For each interval and state, stores the chosen ending border index that
        produced the best cumulative correlation.

    number_of_intervals : int
        Number of warped segments used to divide the signals.

    boundary_indices : list[int]
        Original reference segment boundary indices (unwarped).

    verbose : bool, optional
        If True, prints information messages instead of being silent.

    Returns
    -------
    optimal_warping_path : list[int]
        List of boundary indices in the sample signal after optimal warping,
        starting at 0 and ending at len(sample). Length equals number_of_intervals + 1.
    """

    # Preallocate output list (faster than append in loop)
    optimal_warping_path = [None] * (number_of_intervals + 1)
    optimal_warping_path[0] = boundary_indices[0]
    optimal_warping_path[-1] = boundary_indices[number_of_intervals]

    # Start from the last interval: index of max cumulative correlation
    best_border_index = np.argmax(best_cumulative_correlations[-1])

    # For reverse indexing convenience
    n = number_of_intervals - 1

    # Backtrack through all inner intervals
    for i in range(1, number_of_intervals):
        row = n - i
        # Get selected best border
        best_border = best_end_borders[row, best_border_index]
        optimal_warping_path[i] = int(best_border)
        best_border_index = np.argmax(possible_start_borders[row] == best_border)

    if verbose:
        print(f"[cowarp] optimal_warping_path -> {optimal_warping_path}")

    return optimal_warping_path


def calculate_warped_sample(
        sample,
        num_intervals,
        reference_len,
        boundary_indices,
        optimal_warping_path):
    """
    Construct the final warped sample by resampling each warped segment
    to match the corresponding reference segment length.

    Parameters
    ----------
    sample : ndarray
        Original sample signal to be warped.

    num_intervals : int
        Number of warping segments.

    reference_len : int
        Total length of the reference signal.

    boundary_indices : list[int]
        Segment boundary indices of the reference signal.

    optimal_warping_path : list[int]
        Segment boundary indices of the warped sample (after DP alignment).

    Returns
    -------
    warped_sample : ndarray
        The fully warped sample signal with length == reference_len.
    """
    warped_sample = np.zeros(reference_len)

    for i in range(num_intervals):
        src_start, src_end = optimal_warping_path[i], optimal_warping_path[i + 1]
        dst_start, dst_end = boundary_indices[i], boundary_indices[i + 1]

        segment = sample[src_start:src_end]
        old_len = len(segment)
        new_len = dst_end - dst_start

        if old_len != new_len:
            segment = interpolate_signal(segment, old_len, new_len)

        warped_sample[dst_start:dst_end] = segment

    return warped_sample


def cow_dynamic_no_edges(
        reference,
        sample,
        num_intervals=None,
        interval_length=None,
        slack=None,
        min_interval_length=None,
        return_details=False,
        verbose=False
):
    """
    Perform Correlation Optimized Warping (COW) between two 1D signals.
    The function aligns `sample` to `reference` by
    segment-wise warping with dynamic programming.

    Parameters
    ----------
    reference : array-like
        1D numeric signal used as the reference (target length after warping).
    sample : array-like
        1D numeric signal that will be warped to match the reference.
    num_intervals : int, optional
        Number of warping intervals. Mutually exclusive with `segment_len`.
    interval_length : int, optional
        Desired fixed length of each interval. Used if `num_of_intervals`
        is not provided.
    slack : int, optional
        Maximum number of samples allowed for warping per segment
        (`segment_len ± slack`). If None, determined automatically.
    min_interval_length : int, optional
        Minimum allowed segment length (default = 3 if None).
    return_details : bool, default=False
        If True, return full warping details instead of just warped signal
        and correlation.
    verbose: bool, default=False
        If True, prints internal decisions and computed values.

    Returns
    -------
    warped_sample : np.ndarray
        The warped version of `sample`, aligned to the same length as `reference`.
    final_correlation : float
        Pearson correlation coefficient between `reference` and `warped_sample`.
    If `return_details=True`, returns a dict:
        {
            "warped_sample": np.ndarray,
            "correlation": float,
            "warping_path": list[int],
                List of boundary indices in the sample signal defining the intervals
                that correspond to the fixed reference intervals.
                The length of this list is equal to num_intervals + 1.
            "boundaries": list[int],
                List of fixed interval boundaries in the reference signal.
                The length of this list is equal to num_intervals + 1.
        }

    Raises
    ------
    TypeError
        If `reference` or `sample` cannot be converted to a numeric 1D array.
    ValueError
        * If either signal is shorter than `min_interval_len`
        * If both signals are too short to be warpable (`len < 2*min_interval_len + 1`)
        * If `num_of_intervals`, `segment_len`, or `slack` are invalid (e.g. non-positive)
        * If any input signal contains `NaN` values
    MemoryError
        If resampling or array conversion fails due to insufficient memory.

    Notes
    -----
    - Resamples the shorter signal if only one is warpable.
    - Ensures both signals have equal length before DP warping.
    - Uses dynamic programming to maximize cumulative correlation.
    - Final warped signal is always the same length as `reference`.
    """
    # --- Input validation ---
    reference = validate_input_array(reference)
    sample = validate_input_array(sample)

    ref_len = len(reference)
    samp_len = len(sample)

    min_interval_length = determine_min_interval_length(min_interval_length, verbose)

    # check signal length validity
    if not has_valid_len(ref_len, min_interval_length):
        raise ValueError(f"[cowarp] reference len < min_interval_length")

    if not has_valid_len(samp_len, min_interval_length):
        raise ValueError(f"[cowarp] sample len < min_interval_length")

    # check if signals are warpable
    if not is_warpable_case_no_edges(ref_len, min_interval_length) and \
            not is_warpable_case_no_edges(samp_len, min_interval_length):

        print(f'[cowarp] signal length should be more than or equal to 2*min_interval_len+1 to be warpable')
        raise ValueError("[cowarp] signals are too short for warping.")

    elif not is_warpable_case_no_edges(ref_len, min_interval_length):
        print("[cowarp] reference is not warpable!")
        print(f'signal length should be more than 2*min_interval_len+1 to be warpable')
        print("resampling reference to the length os sample!")
        print()
        if ref_len != samp_len:
            reference = interpolate_signal(reference, ref_len, samp_len)
            reference_len = len(reference)

    elif not is_warpable_case_no_edges(samp_len, min_interval_length):
        print("[cowarp] sample is not warpable")
        print(f'signal length should be more than 2*min_interval_len+1 to be warpable')
        print("resampling sample to the length of reference")
        print()
        if ref_len != samp_len:
            sample = interpolate_signal(sample, samp_len, ref_len)

    # If still mismatched, force equal lengths
    if ref_len != samp_len:
        sample = interpolate_signal(sample, samp_len, ref_len)

    # --- Parameter determination ---
    num_intervals = determine_num_intervals(num_intervals, interval_length, ref_len, min_interval_length, verbose)
    interval_length = ref_len // num_intervals
    slack = determine_slack(slack, interval_length, min_interval_length, verbose)
    boundary_indices = get_boundary_indices(num_intervals, interval_length, ref_len, verbose)

    # --- Dynamic programming matrix ---
    best_cumulative_correlations, possible_start_borders, best_end_borders = \
        fill_correlation_matrix(reference, sample, num_intervals, slack, interval_length,
                                min_interval_length, boundary_indices)

    # --- Optimal path extraction ---
    warping_path = calculate_optimal_warping_path(best_cumulative_correlations,
                                                  possible_start_borders,
                                                  best_end_borders,
                                                  num_intervals, boundary_indices,
                                                  verbose)

    # --- Warp sample using path ---
    warped = calculate_warped_sample(sample, num_intervals, ref_len, boundary_indices, warping_path)

    # One last consistency check
    if len(warped) != ref_len:
        warped = interpolate_signal(warped, samp_len, ref_len)

    # Final similarity
    final_corr = calculate_correlation(reference, warped)
    if verbose:
        print(f"[cowarp] Final correlation: {final_corr:.5f}")

    if return_details:
        return {
            "warped_sample": warped,
            "correlation": final_corr,
            "warping_path": warping_path,
            "boundaries": boundary_indices,
        }
    else:
        return warped, final_corr
