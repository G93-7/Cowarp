import numpy as np
import time
import matplotlib.pyplot as plt

from cowarp import warp

n = 100
reference = np.sin(np.linspace(0, 2 * np.pi, n))
sample = np.sin(np.linspace(0, 2 * np.pi, n) + 0.5)

start_time = time.time()
result_dict = warp(reference, sample,
                   num_intervals=5,
                   slack=8,
                   min_interval_length=2,
                   return_details=True,
                   verbose=True)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time} seconds")

# print(result_dict)
warped_sample = result_dict['warped_sample']
final_corr = result_dict['correlation']

# plots
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)

plt.plot(reference, label='referece')
plt.plot(sample, label='sample')

# plt.title(f'corr= {initial_corr:.2f}')
# plt.xlabel('Index')
# plt.ylabel('Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# plots
plt.subplot(2, 1, 2)
plt.plot(reference, label='referece')
plt.plot(warped_sample, label='warped sample')

plt.title(f'final_corr= {final_corr:.2f}')
# plt.xlabel('Index')
# plt.ylabel('Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.show()
