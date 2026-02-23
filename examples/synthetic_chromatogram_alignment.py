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
    auto_segment=False,
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
