## 📘 Cowarp – Correlation Optimized Warping in Python

**COWarp** is a Python implementation of an alignment method called Correlation Optimized Warping (COW). This technique is widely used in spectroscopy, chromatography, and other analytical fields to align signals that may have local time shifts or distortions. COW aligns one signal (sample) to another (reference) by segmenting the signals and warping segments to maximize correlation.

This is the initial release of the package. It currently includes a single implementation of the COW algorithm, in which the sample signal endpoints remain fixed (i.e., not subject to warping).

![Version](https://img.shields.io/badge/Version-0.1.0-blue)

## 🛠️ Installation

```markdown
pip install cowarp
```

## 🚀 Basic Usage

```markdown
import numpy as np
from cowarp import warp

# Example reference and sample signals
reference = np.sin(np.linspace(0, 2 * np.pi, n))
sample = np.sin(np.linspace(0, 2 * np.pi, n) + 0.5)

# Run Correlation Optimized Warping
aligned_sample, corr = warp(reference, sample, num_intervals=5, slack=8)
```

## ⚙️ Function Signature
```markdown
cow(
    reference,
    sample,
    num_intervals=None,
    interval_length=None,
    slack=None,
    min_interval_length=None,
    return_details=False,
    verbose=False
)
```

## 📚 Parameters

| Parameter             | Type                | Description                                                                               |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------- |
| `reference`           | array-like          | Reference (target) 1D signal to align against.                                               |
| `sample`              | array-like          | Sample signal to be warped and aligned to the reference. Currently supports only 1D signals.                                   |
| `num_intervals`       | int, optional       | Number of intervals to divide the signal into. Mutually exclusive with `interval_length`. |
| `interval_length`     | int, optional       | Length of each interval. Alternative to `num_intervals`.                                  |
| `slack`               | int, optional       | Maximum allowed deviation (shift) in interval boundary positions.                         |
| `min_interval_length` | int, optional       | Minimum length allowed for any interval. Default: 3.                                      |
| `return_details`      | bool, default=False | If True, returns additional details (e.g., warp path).                      |
| `verbose`             | bool, default=False | If True, prints internal decisions and computed values.                            |

## 📤 Returns

Depending on `return_details`:
- If `False` returns a tuple:
`(warped_sample, correlation)`

  - `warped_sample` : np.ndarray
  
    The warped (aligned ) version of `sample`, aligned to the same length as `reference`.

  - `correlation` :     final_correlation : float
 
    Pearson correlation coefficient between `reference` and `warped_sample`.

- If `True` returns a dictionary:

  ```markdown
        {
            "warped_sample": np.ndarray,
            "correlation": float,
            "warping_path": list[int],
            "boundaries": list[int],
        }
  ```
  - `warping_path`
    
    List of boundary indices in the sample signal defining the intervals
    that correspond to the fixed reference intervals.
    The length of this list is equal to num_intervals + 1.

  - `boundaries` 

    List of fixed interval boundaries in the reference signal. The length of this list is equal to num_intervals + 1.

## 🧠 Example

```markdown
"""
Example: Aligning Chromatographic Signals using COW

This example demonstrates how to use the COW (Correlation Optimized Warping)
algorithm to align two chromatographic signals. The sample chromatogram
contains nonlinear time shifts and noise relative to the reference.

"""

import numpy as np
import matplotlib.pyplot as plt
from cowarp import warp


def generate_chromatogram(num_points=400, random_shift=False, noise_level=0.01):
    x = np.linspace(0, 100, num_points)

    # Define ideal peaks
    peaks = [15, 35, 55, 75]
    widths = [2.5, 3.5, 2.5, 3.0]
    amplitudes = [1.0, 0.9, 1.2, 0.8]

    y = np.zeros_like(x)
    for i, (mu, sigma, amp) in enumerate(zip(peaks, widths, amplitudes)):
        if random_shift:
            mu += np.random.uniform(-3.0, 3.0)  # stronger shift
        y += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Add small baseline and noise
    y += 0.05 + noise_level * np.random.randn(num_points)
    return x, y


# --- Generate reference and sample chromatograms ---

# Reference chromatogram
x_ref, reference = generate_chromatogram(num_points=500, random_shift=False, noise_level=0.005)

# Sample chromatogram (more strongly shifted and noisier)
x_samp, sample = generate_chromatogram(num_points=480, random_shift=True, noise_level=0.02)

# --- Perform COW alignment ---
result_dict = warp(
    reference,
    sample,
    num_intervals=10,
    slack=25,
    min_interval_length=4,
    return_details=True,
    verbose=True
)

warped_sample = result_dict['warped_sample']
final_corr = result_dict['correlation']
warping_path = result_dict['warping_path']
boundaries = result_dict['boundaries']

print(f"Final correlation after warping: {final_corr:.4f}")

# --- Plot results ---
plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.plot(x_ref, reference, label="Reference", linewidth=2)
plt.plot(x_samp, sample, label="Sample (before warping)", linestyle="--")
plt.title("Before COW Alignment")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_ref, reference, label="Reference", linewidth=2)
plt.plot(np.linspace(0, 100, len(warped_sample)), warped_sample, label="Sample (after warping)", linestyle="--")
plt.title(f"After COW Alignment (Correlation = {final_corr:.4f})")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```

The resulting plot is given below.

![COW Chromatogram Alignment](images/chromatogram_alignment.jpeg)

## 🧾 License

This project is licensed under the MIT License.

## ✍️ Author

Guram Chaganava

Electrical & Electronics Engineer | Machine Learning Enthusiast




