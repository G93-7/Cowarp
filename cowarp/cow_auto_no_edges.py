# Correlation Optimized Warping with Automatic Segmentation (ASCOW) - dynamic programming implementation.

# The author refers to this version of COW as a *auto* because
# it performs automatic segmentation - determining segment boundaries

# The term *no edges* means that the sample signal edges (first and last points)
# are not warped — they remain fixed during the alignment process.

# Works fully with NumPy arrays.
# Symmetric structure, increasing number of possible boundary positions from edge to center.
# Vectorized solution for speed-up.

import numpy as np

# constants
STATIONARY_POINTS = 'stationary_points'
INFLECTION_POINTS = 'inflection_points'
PEAK_BOUNDARIES = 'peak_boundaries'

# min signal length for segmentation modes
STATIONARY_POINTS_MIN_LEN = 2
INFLECTION_POINTS_MIN_LEN = 3

# min signal length dictionary
min_len = {
    STATIONARY_POINTS: STATIONARY_POINTS_MIN_LEN,
    INFLECTION_POINTS: INFLECTION_POINTS_MIN_LEN,
    PEAK_BOUNDARIES: STATIONARY_POINTS_MIN_LEN
}

# filter codes
MOVING_AVERAGE_FILTER = 'moving_average'
GAUSSIAN_FILTER = 'gaussian'
NO_FILTER = 'no_filter'


# functions
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


def validate_int_input(input):
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


def validate_float_input(input):
    """
    Validate and convert a single numeric input to a float.

    Accepts:
        • Python int or float
        • NumPy scalar (0-D array)
        • Raises an error if input is non-numeric, NaN, or not scalar

    Conversion rules:
        • Integer inputs are converted to float
        • Float inputs are returned unchanged
        • NumPy scalar values are converted to Python scalars first
        • Non-scalar NumPy arrays are rejected

    Parameters
    ----------
    input : int, float, numpy scalar, or array-like
        Input value expected to represent a single numeric quantity.

    Returns
    -------
    float
        Floating-point value after validation.

    Raises
    ------
    ValueError
        If input is an array, non-numeric, or NaN.
    """

    # Handle NumPy scalar arrays (0D only)
    if isinstance(input, np.ndarray):
        if input.ndim != 0:
            raise ValueError("[cowarp] Input must be a scalar, not an array.")
        input = input.item()

    # Integer type → convert to float
    if isinstance(input, (int, np.integer)):
        return float(input)

    # Float type → validate and return
    if isinstance(input, (float, np.floating)):
        if np.isnan(input):
            raise ValueError("[cowarp] Input cannot be NaN.")
        return float(input)

    # Anything else → reject
    raise ValueError("[cowarp] Input must be a numeric scalar (int or float).")


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

    min_interval_length = validate_int_input(min_interval_length)

    if min_interval_length < 2:
        raise ValueError(f"[cowarp] min_interval_length={min_interval_length} is invalid; must be >= 2")

    if verbose:
        print(f"[cowarp] Using min_interval_length = {min_interval_length}")

    return min_interval_length


def determine_deformation_coeff(deformation_coeff, verbose=False):
    """
    Validate and determine the deformation coefficient for COW.

    The deformation coefficient controls the allowable boundary
    flexibility during warping and must satisfy:

        0 ≤ deformation_coeff ≤ 1

    If `deformation_coeff` is None, a default value of 0.3 is used.

    Parameters
    ----------
    deformation_coeff : float or None
        User-specified deformation coefficient. If None,
        a default value of 0.3 is assigned.
    verbose : bool, optional
        If True, prints the selected deformation coefficient.

    Returns
    -------
    float
        Validated deformation coefficient within [0, 1].

    Raises
    ------
    ValueError
        If deformation_coeff is outside the allowed range
        or cannot be converted to a valid float.
    """
    min_deformation_coeff = 0
    max_deformation_coeff = 1

    if deformation_coeff is None:
        deformation_coeff = 0.3
        if verbose:
            print(f"[cowarp] deformation_coeff not provided, using default={deformation_coeff}")

    deformation_coeff = validate_float_input(deformation_coeff)

    # Check validity range
    if deformation_coeff < min_deformation_coeff:
        raise ValueError(
            f"[cowarp] deformation_coeff={deformation_coeff} is below the minimum allowed ({min_deformation_coeff})"
        )
    if deformation_coeff > max_deformation_coeff:
        raise ValueError(
            f"[cowarp] deformation_coeff={deformation_coeff} exceeds the maximum allowed ({max_deformation_coeff})"
        )

    if verbose:
        print(f"[cowarp] Using deformation_coeff = {deformation_coeff}")

    return deformation_coeff


def determine_min_signal_len(min_interval_len, segmentation_type):
    """
    Determine the minimum required signal length.

    The function returns the larger of:
        - The minimum signal length required by the given segmentation type
        - The provided minimum interval length

    This ensures that the signal length is always compatible with both
    the segmentation constraints and the minimum interval size.

    Parameters
    ----------
    min_interval_len : int
        User-defined minimum allowed interval length.
    segmentation_type : int
        Segmentation mode key used to retrieve the baseline minimum
        signal length from `min_len`.

    Returns
    -------
    int
        Minimum required signal length.
    """
    min_signal_len = min_len[segmentation_type]
    if min_interval_len > min_signal_len:
        return min_interval_len
    return min_signal_len


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


def moving_average(signal, window_size):
    """
    Compute the moving average of a 1D signal.

    The function applies a simple sliding-window average using
    convolution with a uniform kernel of size `window_size`.
    The output is computed in 'valid' mode, meaning only positions
    where the full window overlaps the signal are returned.

    Parameters
    ----------
    signal : array-like
        Input 1D signal.
    window_size : int
        Size of the averaging window. Must be a positive integer.

    Returns
    -------
    numpy.ndarray
        Smoothed signal of length len(signal) - window_size + 1.

    Notes
    -----
    - No padding is applied.
    - Larger window sizes produce stronger smoothing.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')


# TODO: remove scipy dependence
from scipy.ndimage import gaussian_filter1d


def gaussian_filter(signal, sigma):
    """
    Apply one-dimensional Gaussian smoothing to a signal.

    The function smooths the input signal using a Gaussian kernel
    with standard deviation `sigma`, implemented via
    `scipy.ndimage.gaussian_filter1d`.

    Parameters
    ----------
    signal : array-like
        Input 1D signal.
    sigma : float
        Standard deviation of the Gaussian kernel. Larger values
        produce stronger smoothing.

    Returns
    -------
    numpy.ndarray
        Smoothed signal of the same length as the input.

    Notes
    -----
    - Unlike a simple moving average, Gaussian smoothing applies
      weighted averaging with weights following a normal distribution.
    - Boundary handling follows the default behavior of
      `gaussian_filter1d`.
    """
    return gaussian_filter1d(signal, sigma=sigma)


#  dictionary of filter functions
filter_func_dict = {
    GAUSSIAN_FILTER: gaussian_filter,
    MOVING_AVERAGE_FILTER: moving_average,
    NO_FILTER: None
}


def filter_signal(signal, filter_code=MOVING_AVERAGE_FILTER, filter_param=3):
    """
    Apply a selected smoothing filter to a signal.

    The function dispatches the filtering operation based on
    `filter_code`, using the corresponding function stored in
    `filter_func_dict`.

    Parameters
    ----------
    signal : array-like
        Input 1D signal to be filtered.
    filter_code : hashable, optional
        Key identifying the filter type. Must exist in
        `filter_func_dict`. Defaults to MOVING_AVERAGE_FILTER.
    filter_param : int or float, optional
        Parameter passed to the selected filter function.
        For example:
            - window size (moving average)
            - sigma (Gaussian filter)
        Defaults to 3.

    Returns
    -------
    numpy.ndarray
        Filtered signal.

    Raises
    ------
    KeyError
        If `filter_code` is not found in `filter_func_dict`.
    """
    return filter_func_dict[filter_code](signal, filter_param)


def get_stationery_points(y):
    """
    Detect stationary points in a 1D signal.

    A stationary point is identified where the sign of the first
    discrete difference changes between consecutive samples,
    indicating a local maximum or minimum.

    Parameters
    ----------
    y : array-like
        Input 1D signal.

    Returns
    -------
    list of int
        Indices of detected stationary points.

    Notes
    -----
    - The method uses sign changes in successive differences:
          sign(y[i+1] - y[i])
    """
    stationary_points = []
    previous_sign = np.sign(y[1] - y[0])
    for i in range(1, len(y) - 1):
        new_sign = np.sign(y[i + 1] - y[i])
        if new_sign != previous_sign:
            stationary_points.append(i)
        previous_sign = new_sign
    return stationary_points


def get_inflection_points(y):
    """
    Detect inflection points in a 1D signal.

    Inflection points are identified as stationary points of the
    first discrete derivative. The function first computes the
    forward difference:

        dy[i] = y[i+1] - y[i]

    and then detects sign changes in `dy` using `get_stationery_points`.

    Parameters
    ----------
    y : array-like
        Input 1D signal.

    Returns
    -------
    list of int
        Indices corresponding to inflection points (relative to
        the derivative signal).

    Notes
    -----
    - This method approximates inflection points via sign changes
      in the second discrete difference.
    """
    dy = []
    for i in range(len(y) - 1):
        dy.append(y[i + 1] - y[i])
    return get_stationery_points(dy)


def find_peaks(r):
    """
    Detect local maxima (peaks) in a 1D signal.

    A peak is defined as a point whose value is strictly greater
    than its immediate neighbors:

        r[i] > r[i-1] and r[i] > r[i+1]

    Parameters
    ----------
    r : array-like
        Input 1D signal.

    Returns
    -------
    list of int
        Indices of detected peaks.

    """
    peaks = []
    N = len(r)
    for i in range(1, N - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            peaks.append(i)
    return peaks


def get_peak_boundaries(r, peaks):
    """
    Determine the left boundaries of peaks in a 1D signal.

    For each peak index in `peaks`, the function searches backward
    until the signal starts increasing, marking the left boundary of
    the peak. The last element of `boundaries` is set to the end of
    the signal.

    Parameters
    ----------
    r : array-like
        Input 1D signal.
    peaks : list of int
        Indices of detected peaks in the signal.

    Returns
    -------
    list of int
        Indices representing the left boundaries of each peak.
        The last element is the signal length.

    Notes
    -----
    - The function only computes the **left boundary** for each peak.
    - Boundaries are determined by scanning backward until the slope
      becomes negative.
    - This method assumes peaks are separated and does not handle
      overlapping peaks.
    """
    last_index = len(r)
    boundaries = []
    for p in peaks:
        i = p
        while i - 1 >= 0 and r[i - 1] <= r[i]:
            i -= 1
        boundaries.append(i)
    boundaries.append(last_index)
    return boundaries


def get_boundary_indices(reference, minimum_interval_length, segmentation_type=STATIONARY_POINTS, verbose=False):
    """
    Compute boundary indices for segmenting a 1D signal.

    Depending on the chosen `segmentation_type`, boundaries are
    identified based on stationary points, inflection points, or
    peak boundaries. After initial detection, boundaries that are
    closer together than `minimum_interval_length` are removed
    to enforce a minimum segment size.

    Parameters
    ----------
    reference : array-like
        Input 1D signal used for segmentation.
    minimum_interval_length : int
        Minimum allowed distance between consecutive boundaries.
    segmentation_type : str, optional
        Segmentation strategy to use:
        - STATIONARY_POINTS: boundaries at stationary points
        - INFLECTION_POINTS: boundaries at inflection points
        - PEAK_BOUNDARIES: boundaries at peak boundaries
        Defaults to STATIONARY_POINTS.
    verbose : bool, optional
        If True, prints the resulting boundary indices. Defaults to False.

    Returns
    -------
    list of int
        Sorted list of boundary indices including 0 and the last
        index of the signal.

    Notes
    -----
    - Ensures no two consecutive boundaries are closer than
      `minimum_interval_length`.
    - The first boundary is always 0 and the last boundary is
      always `len(reference)`.
    - The exact locations of boundaries depend on the chosen
      segmentation method.
    """
    boundary_indices = []
    boundary_indices_initial = [0]

    if segmentation_type == STATIONARY_POINTS:
        for index in get_stationery_points(reference):
            boundary_indices_initial.append(index)

    elif segmentation_type == INFLECTION_POINTS:
        for index in get_inflection_points(reference):
            boundary_indices_initial.append(index)

    elif segmentation_type == PEAK_BOUNDARIES:
        for index in get_peak_boundaries(reference, find_peaks(reference)):
            boundary_indices_initial.append(index)

    boundary_indices_initial.append(len(reference))

    # remove indices closer than minimum_interval_length
    boundary_indices.append(boundary_indices_initial[0])
    for i in range(len(boundary_indices_initial) - 2):
        if (boundary_indices_initial[i + 1] - boundary_indices_initial[i]) >= minimum_interval_length:
            boundary_indices.append(boundary_indices_initial[i + 1])

    if (boundary_indices_initial[len(boundary_indices_initial) - 1] - boundary_indices[
        len(boundary_indices) - 1]) >= minimum_interval_length:
        boundary_indices.append(boundary_indices_initial[len(boundary_indices_initial) - 1])
    else:
        boundary_indices.pop()
        boundary_indices.append(boundary_indices_initial[len(boundary_indices_initial) - 1])

    if verbose:
        print(f'[cowarp] boundary_indices: {boundary_indices}')
    return boundary_indices


def get_max_num_of_boundary_states(deformation_coeff, num_intervals, min_interval_len, boundary_indices):
    """
    Compute the maximum number of possible boundary states for Correlation Optimized Warping (COW).

    This function estimates how many different positions each boundary
    can take, given the deformation coefficient, interval limits, and
    detected boundary indices. The maximum across all boundaries is returned.

    Parameters
    ----------
    deformation_coeff : float
        Coefficient controlling allowed deformation:
        - 1 means full interval flexibility using min_interval_len
        - <1 scales allowable boundary positions proportionally
    num_intervals : int
        Total number of warping intervals.
    min_interval_len : int
        Minimum allowed length of each interval.
    boundary_indices : list of int
        List of boundary indices including start and end of the signal.

    Returns
    -------
    int
        Maximum number of possible positions (states) among all boundaries.

    Notes
    -----
    - For deformation_coeff == 1, the calculation ensures boundaries
      respect the minimum interval length.
    - For deformation_coeff < 1, the number of states is scaled by
      the deformation coefficient and rounded down to integer.
    - Only internal boundaries (excluding first and last) are considered.
    """
    max_num_boundary_states = 1

    if deformation_coeff == 1:
        for boundary in range(1, num_intervals):
            boundary_states = boundary_indices[boundary + 1] - boundary_indices[boundary - 1] - \
                              2 * min_interval_len + 1
            if boundary_states > max_num_boundary_states:
                max_num_boundary_states = boundary_states
    else:
        for boundary in range(1, num_intervals):
            boundary_states = int((boundary_indices[boundary + 1] - boundary_indices[boundary]) * deformation_coeff) + \
                              int((boundary_indices[boundary] - boundary_indices[boundary - 1]) * deformation_coeff) + 1
            if boundary_states > max_num_boundary_states:
                max_num_boundary_states = boundary_states

    return max_num_boundary_states


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


def fill_correlation_matrix(reference, sample, num_intervals,
                            min_interval_len, boundary_indices,
                            deformation_coeff):
    """
    Fill the dynamic programming (DP) correlation matrices for
    Correlation Optimized Warping (COW).

    The function computes cumulative correlation scores between
    corresponding reference and sample segments under boundary
    deformation constraints. Segments are matched in reverse order
    (last to first), and for each possible boundary state the best
    cumulative correlation is stored.

    Parameters
    ----------
    reference : np.ndarray
        Reference signal (1D array).
    sample : np.ndarray
        Sample signal (1D array) to be warped.
    num_intervals : int
        Number of warping intervals.
    min_interval_len : int
        Minimum allowed length of each interval.
    boundary_indices : list of int
        Reference boundary indices defining segmentation
        (length = num_intervals + 1).
    deformation_coeff : float
        Deformation coefficient controlling allowed boundary
        movement. If equal to 1, constraints are based strictly on
        `min_interval_len`. Otherwise, allowable shifts are scaled
        proportionally to adjacent interval lengths.

    Returns
    -------
    best_cumulative_correlations : np.ndarray
        Matrix (num_intervals × max_states) containing the best
        cumulative correlation values for each interval and
        boundary state.
    possible_start_borders : np.ndarray
        Matrix storing candidate start borders for each interval.
    best_end_borders : np.ndarray
        Matrix storing the optimal end border corresponding to each
        boundary state (used for backtracking).

    Notes
    -----
    - Pearson correlation is computed between centered segments.
    - Zero-norm segments are handled explicitly:
        * both zero → correlation = 1
        * one zero  → correlation = 0
    - Sample segments are interpolated to match the reference
      segment length when necessary.
    - The algorithm proceeds from the last segment to the first,
      accumulating optimal correlations via dynamic programming.
    """

    max_num_border_states = get_max_num_of_boundary_states(deformation_coeff, num_intervals,
                                                           min_interval_len, boundary_indices)

    # Initialize DP matrices
    best_cumulative_correlations = np.full((num_intervals, max_num_border_states), -1.0, dtype=float)
    possible_start_borders = np.full((num_intervals, max_num_border_states), -1, dtype=int)
    best_end_borders = np.full((num_intervals - 1, max_num_border_states), -1, dtype=int)

    # Precompute reference segments and norms
    ref_segments = []
    ref_norms = []
    for k in range(num_intervals):
        seg = reference[boundary_indices[k]:boundary_indices[k + 1]].astype(float)
        seg_cent = seg - seg.mean()
        ref_segments.append(seg_cent)
        ref_norms.append(np.linalg.norm(seg_cent))

    last_segment_len = len(ref_segments[-1])

    # Helper function: compute correlation with zero-norm handling
    def compute_corr(a, b, norm_a, norm_b):
        if norm_a == 0 and norm_b == 0:
            return 1.0
        elif norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    # Last segment
    start_border_idx = 0

    if deformation_coeff == 1:
        interval_start = boundary_indices[num_intervals - 2] + min_interval_len
        interval_end = boundary_indices[num_intervals] - min_interval_len + 1
    else:
        lower_slack = (boundary_indices[num_intervals - 1] - boundary_indices[num_intervals - 2]) * deformation_coeff
        lower_slack = int(lower_slack)

        upper_slack = (boundary_indices[num_intervals] - boundary_indices[num_intervals - 1]) * deformation_coeff
        upper_slack = int(upper_slack)

        interval_start = boundary_indices[num_intervals - 1] - lower_slack
        interval_end = boundary_indices[num_intervals - 1] + upper_slack + 1

    ref_cent = ref_segments[-1]
    ref_norm = ref_norms[-1]

    for new_start in range(interval_start, interval_end):
        new_len = boundary_indices[-1] - new_start
        if new_len < min_interval_len:
            continue

        sample_seg = sample[new_start:boundary_indices[-1]]
        sample_seg_len = len(sample_seg)

        # Interpolation if needed
        if sample_seg_len != last_segment_len:
            sample_seg = interpolate_signal(sample_seg, sample_seg_len, last_segment_len)

        sample_cent = sample_seg - sample_seg.mean()
        sample_norm = np.linalg.norm(sample_cent)
        best_cumulative_correlations[0, start_border_idx] = compute_corr(ref_cent, sample_cent, ref_norm, sample_norm)
        possible_start_borders[0, start_border_idx] = new_start
        start_border_idx += 1

    # Inner segments
    for interval_idx in range(1, num_intervals - 1):
        ref_cent = ref_segments[num_intervals - interval_idx - 1]
        ref_norm = ref_norms[num_intervals - interval_idx - 1]
        segment_len = len(ref_cent)

        first_end_border = possible_start_borders[interval_idx - 1][0]

        start_border_idx = 0

        if deformation_coeff == 1:
            interval_start = boundary_indices[num_intervals - interval_idx - 2] + min_interval_len
            interval_end = boundary_indices[num_intervals - interval_idx] - min_interval_len + 1
        else:
            lower_slack = (boundary_indices[num_intervals - interval_idx - 1] -
                           boundary_indices[num_intervals - interval_idx - 2]) * deformation_coeff
            lower_slack = int(lower_slack)

            upper_slack = (boundary_indices[num_intervals - interval_idx] -
                           boundary_indices[num_intervals - interval_idx - 1]) * deformation_coeff
            upper_slack = int(upper_slack)

            interval_start = boundary_indices[num_intervals - interval_idx - 1] - lower_slack
            interval_end = boundary_indices[num_intervals - interval_idx - 1] + upper_slack + 1

        for new_start in range(interval_start, interval_end):

            end_indices = possible_start_borders[interval_idx - 1]
            lengths = end_indices - new_start
            mask = (lengths >= min_interval_len)

            if not np.any(mask):
                continue

            valid_end_indices = end_indices[mask]
            valid_lengths = lengths[mask]
            num_of_segments = len(valid_end_indices)

            seg_interp = np.full((num_of_segments, segment_len), np.nan, dtype=float)
            for i in range(num_of_segments):
                seg_interp[i] = interpolate_signal(sample[new_start:valid_end_indices[i]],
                                                   valid_lengths[i], segment_len)

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
            corrs += best_cumulative_correlations[interval_idx - 1, valid_end_indices - first_end_border]

            best_idx = np.argmax(corrs)
            best_val = corrs[best_idx]
            best_end = valid_end_indices[best_idx]

            possible_start_borders[interval_idx, start_border_idx] = new_start
            best_end_borders[interval_idx - 1, start_border_idx] = best_end
            best_cumulative_correlations[interval_idx, start_border_idx] = best_val

            start_border_idx += 1

    # first segment
    ref_cent = ref_segments[0]
    ref_norm = ref_norms[0]
    segment_len = len(ref_cent)

    first_end_border = possible_start_borders[num_intervals - 2][0]

    end_indices = possible_start_borders[num_intervals - 2]
    mask = (end_indices >= min_interval_len)
    valid_end_indices = end_indices[mask]
    num_of_segments = len(valid_end_indices)

    seg_interp = np.full((num_of_segments, segment_len), np.nan, dtype=float)
    for i in range(num_of_segments):
        seg_interp[i] = interpolate_signal(sample[0:valid_end_indices[i]],
                                           valid_end_indices[i], segment_len)

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
    corrs += best_cumulative_correlations[num_intervals - 2, valid_end_indices - first_end_border]

    best_idx = np.argmax(corrs)
    best_val = corrs[best_idx]
    best_end = end_indices[best_idx]

    possible_start_borders[num_intervals - 1, 0] = 0
    best_end_borders[num_intervals - 2, 0] = best_end
    best_cumulative_correlations[num_intervals - 1, 0] = best_val

    # print(f'[cowarp] best_cumulative_correlations: {best_cumulative_correlations}')
    # print(f'[cowarp] possible_start_borders: {possible_start_borders}')
    # print(f'[cowarp] best_end_borders: {best_end_borders}')

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
    best_cumulative_correlations : ndarray, shape (num_intervals, max_states)
        Matrix of maximum cumulative correlations at each interval and state.

    possible_start_borders : ndarray, shape (num_intervals, max_states)
        For each interval and state, stores the valid starting index of the sample segment.

    best_end_borders : ndarray, shape (num_intervals - 1, max_states)
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


def calculate_warped_sample(sample,
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


def cow_auto_no_edges(
        reference,
        sample,
        segmentation_type,
        deformation_coeff,
        filter_func_code,
        filter_func,
        filter_func_params,
        process_filtered_signals,
        min_interval_length,
        return_details,
        verbose
):
    """
    Perform Correlation Optimized Warping (COW) with Automatic Segmentation (ASCOW).

    The function aligns `sample` to `reference` using segmentation-based
    dynamic programming. Segment boundaries are determined automatically
    from the reference signal (stationary points, inflection points,
    or peak boundaries). Boundary flexibility is controlled by the
    deformation coefficient.

    Parameters
    ----------
    reference : array-like
        1D reference signal (target length after warping).
    sample : array-like
        1D signal to be warped to match `reference`.
    segmentation_type : str, optional
        Strategy for determining reference boundaries. Can be one of the following:
            - 'stationary_points'
            - 'inflection_points'
            - 'peak_boundaries'
    min_interval_length : int, optional
        Minimum allowed interval length. If None, determined automatically.
    deformation_coeff : float, optional
        Boundary deformation coefficient in [0, 1].
        Controls allowable boundary movement.
        If None, a default value is used.
    filter_func_code : str, optional
        Predefined filter identifier. Can be one of the following:
        - 'moving_average'
        - 'gaussian'
        - 'no_filter'
    filter_func : callable, optional
        Custom filtering function. If provided, overrides `filter_func_code`.
    filter_func_params : int or float, optional
        Parameter passed to the selected filter function
        (e.g., window size or sigma).
    process_filtered_signals : bool, default=False
        If True, segmentation and warping are performed on filtered
        signals. Otherwise, filtering is used only to determine boundaries.
    return_details : bool, default=False
        If True, returns full warping details instead of only
        (warped_signal, correlation).
    verbose : bool, default=False
        If True, prints intermediate computation details.

    Returns
    -------
    warped_sample : np.ndarray
        Warped version of `sample` aligned to the length of `reference`.
    final_correlation : float
        Pearson correlation coefficient between `reference`
        and `warped_sample`.

    If `return_details=True`, returns:
        {
            "warped_sample": np.ndarray,
            "correlation": float,
            "warping_path": list[int],
            "boundaries": list[int],
        }

    Raises
    ------
    ValueError
        If signals are too short, not warpable, or contain invalid values.
    TypeError
        If inputs cannot be converted to numeric 1D arrays.

    Notes
    -----
    - Automatically determines the number of intervals from
      segmentation boundaries.
    - Ensures both signals have equal length before warping.
    - Uses dynamic programming to maximize cumulative segment-wise
      correlation.
    - Final warped signal always matches the length of `reference`.
    """
    # --- Input validation ---
    reference = validate_input_array(reference)
    sample = validate_input_array(sample)

    ref_len = len(reference)
    samp_len = len(sample)

    min_interval_length = determine_min_interval_length(min_interval_length, verbose)
    min_signal_len = determine_min_signal_len(min_interval_length, segmentation_type)
    deformation_coeff = determine_deformation_coeff(deformation_coeff, verbose)

    # check signal length validity
    if not has_valid_len(ref_len, min_signal_len):
        raise ValueError(f"[cowarp] reference len < min_signal_len")

    if not has_valid_len(samp_len, min_signal_len):
        raise ValueError(f"[cowarp] sample len < min_signal_len")

    # check if signals are warpable
    if not is_warpable_case_no_edges(ref_len, min_interval_length) and \
            not is_warpable_case_no_edges(samp_len, min_interval_length):

        print(f'[cowarp] signal length should be more than or equal to '
              f'2*min_interval_len+1={2 * min_interval_length + 1} to be warpable')
        raise ValueError("[cowarp] signals are too short for warping.")

    elif not is_warpable_case_no_edges(ref_len, min_interval_length):
        print("[cowarp] reference is not warpable!")
        print(f'signal length should be more than '
              f'2*min_interval_len+1={2 * min_interval_length + 1} to be warpable')
        print("resampling reference to the length of sample!")
        print()
        if ref_len != samp_len:
            reference = interpolate_signal(reference, ref_len, samp_len)
            ref_len = len(reference)

    elif not is_warpable_case_no_edges(samp_len, min_interval_length):
        print("[cowarp] sample is not warpable")
        print(f'signal length should be more than '
              f'2*min_interval_len+1={2 * min_interval_length + 1} to be warpable')
        print("resampling sample to the length of reference")
        print()
        if ref_len != samp_len:
            sample = interpolate_signal(sample, samp_len, ref_len)

    # If still mismatched, force equal lengths
    if ref_len != samp_len:
        sample = interpolate_signal(sample, samp_len, ref_len)

    # filter signals
    if filter_func:

        # if original, unfiltered signals must be processed
        if not process_filtered_signals:
            reference_copy = reference.copy()
            reference_copy = filter_func(reference_copy, filter_func_params)
            reference_copy = interpolate_signal(reference_copy, len(reference_copy), ref_len)
            boundary_indices = get_boundary_indices(reference_copy, min_interval_length, segmentation_type)

        # if filtered signals must be processed
        else:
            reference = filter_func(reference, filter_func_params)
            sample = filter_func(sample, filter_func_params)

            ref_len = len(reference)
            samp_len = len(sample)
            if samp_len != ref_len:
                sample = interpolate_signal(sample, samp_len, ref_len)
            boundary_indices = get_boundary_indices(reference, min_interval_length, segmentation_type)

    elif isinstance(filter_func_code, str) and filter_func_code in filter_func_dict:

        # if one of the filter flags is passed
        if filter_func_code != NO_FILTER:

            # if original, unfiltered signals must be processed
            if not process_filtered_signals:
                reference_copy = reference.copy()
                reference_copy = filter_signal(reference_copy, filter_code=filter_func_code,
                                               filter_param=filter_func_params)
                reference_copy = interpolate_signal(reference_copy, len(reference_copy), ref_len)
                boundary_indices = get_boundary_indices(reference_copy, min_interval_length, segmentation_type)

            # if filtered signals must be processed
            else:
                reference = filter_signal(reference, filter_code=filter_func_code,
                                          filter_param=filter_func_params)
                sample = filter_signal(sample, filter_code=filter_func_code, filter_param=filter_func_params)

                ref_len = len(reference)
                samp_len = len(sample)

                if samp_len != ref_len:
                    sample = interpolate_signal(sample, samp_len, ref_len)
                boundary_indices = get_boundary_indices(reference, min_interval_length, segmentation_type)

        # if 'no filter' flag is passed
        else:
            boundary_indices = get_boundary_indices(reference, min_interval_length, segmentation_type)

    if verbose:
        print(f'[cowarp] boundary_indices: {boundary_indices}')
    num_intervals = len(boundary_indices) - 1

    if num_intervals == 1:

        # Final similarity
        final_corr = calculate_correlation(reference, sample)

        if verbose:
            print(f"[cowarp] Final correlation: {final_corr:.5f}")

        if return_details:
            return {
                "warped_sample": sample,
                "correlation": final_corr,
                "warping_path": boundary_indices,
                "boundaries": boundary_indices,
            }
        else:
            return sample, final_corr

    # --- Dynamic programming matrix ---
    best_cumulative_correlations, possible_start_borders, best_end_borders = \
        fill_correlation_matrix(reference, sample, num_intervals, min_interval_length,
                                boundary_indices, deformation_coeff)

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
