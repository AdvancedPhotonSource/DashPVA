import numpy as np
import cupy as cp
import time

#CPU
a = np.random.rand(10000, 10000)

# Vectorized operation with NumPy: adding 1 to each element where value is 0.5
t1 = time.time()
a[a == 0.5] += 1
t2 = time.time()
vectorized_time = t2 - t1

a = np.random.rand(10000, 10000)

# # Loop operation with NumPy
# t1 = time.time()
# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         if a[i, j] == 0.5:
#             a[i, j] += 1
# t2 = time.time()
# loop_time = t2 - t1


with cp.cuda.Device(0):
    #GPU
    a_gpu = cp.random.rand(10000, 10000)
    # Verify that the array is on the correct device
    assert a_gpu.device.id == 0, "CuPy is not using the correct CUDA device"
    # Vectorized operation with CuPy
    t1 = time.time()
    a_gpu[a_gpu == 0.5] += 1
    t2 = time.time()
    vectorized_gpu_time = t2 - t1

    a_gpu = cp.random.rand(10000, 10000)    
    
    # # Loop operation with CuPy
    # t1 = time.time()
    # for i in range(a_gpu.shape[0]):
    #     for j in range(a_gpu.shape[1]):
    #         if a_gpu[i, j] == 0.5:
    #             a_gpu[i, j] += 1
    # t2 = time.time()
    # loop_gpu_time = t2 - t1

print("Vectorized Time (CPU, NumPy):", vectorized_time)
# print("Loop Time (CPU, NumPy):", loop_time)
print("Vectorized Time (GPU, CuPy, Device 0):", vectorized_gpu_time)
# print("Loop Time (GPU, CuPy, Device 0):", loop_gpu_time)
