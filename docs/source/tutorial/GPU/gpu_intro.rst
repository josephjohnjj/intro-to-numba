CUDA and Numba
--------------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min

        **Objectives:**
            #. Learn how target GPUs using Numba.

Kernel Function
****************

A kernel function is a GPU function called from CPU code that cannot return values directly.
It also define how GPU threads hierarchy (threads, blocks and grids) is used. 

..  code-block:: python
    :emphasize-lines: 1
    :linenos:

    @cuda.jit
    def polar_to_cartesian(rho, theta):
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)

Device Functions
****************

Device functions are used to perform computations on the GPU, and they can be invoked from within 
other device functions or kernels. Unlike a kernel function, a device function can return a value
like normal functions.


..  code-block:: python
    :emphasize-lines: 1
    :linenos:

    @cuda.jit(device=True) 
    def polar_to_cartesian(rho, theta):
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return x, y

`@vectorize` can also target GPU.

..  code-block:: python
    :emphasize-lines: 1
    :linenos:

    @cuda.jit(device=True)
    def polar_to_cartesian(rho, theta):
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return x, y  

    @vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
    def polar_distance(rho1, theta1, rho2, theta2):
        x1, y1 = polar_to_cartesian(rho1, theta1)
        x2, y2 = polar_to_cartesian(rho2, theta2)

        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


Thread Indexing
****************

When launching a kernel, you should also specify the thread arrangements.

..  code-block:: python
    :linenos:

    @cuda.jit
    def increment_a_2D_array(an_array):
        x, y = cuda.grid(2)
        if x < an_array.shape[0] and y < an_array.shape[1]:
           an_array[x, y] += 1

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    increment_a_2D_array[blockspergrid, threadsperblock](an_array)

You can learn more about thread indexing in the tutorial 
`Introduction to Parallel Programming Using Python <https://intro-to-parallel-programming.readthedocs.io/en/latest>`_ .
    

Memory management
******************

Although Numba can automatically transfer NumPy arrays to the device and back, we can prevent 
unnecessary transfers by manually controlling the transfer process.

Host to device copy:

..  code-block:: python
    :linenos:

    data_cpu = np.arange(10)
    data_gpu = cuda.to_device(data_cpu)

Device to host copy:

..  code-block:: python
    :linenos:

    data_cpu = data_gpu.copy_to_host()


or ...

..  code-block:: python
    :linenos:

    data_gpu.copy_to_host(data_cpu)

Streams
*******

Streams are sequences of operations that are executed in order on the GPU. Operations in different 
streams can run concurrently, allowing for parallel execution and better utilization of GPU resources.
CUDA streams in Numba allow you to manage and execute multiple tasks concurrently on a GPU, enhancing 
performance by overlapping computation and data transfer operations. 

..  code-block:: python
    :linenos:

    from numba import cuda
    import numpy as np

    # Define a simple kernel function
    @cuda.jit
    def add_kernel(a, b, c):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x

        pos = tx + ty * bw

        if pos < a.size:
            c[pos] = a[pos] + b[pos]

    # Create two streams
    stream1 = cuda.stream()
    stream2 = cuda.stream()

    # Initialize data
    size = 1000000
    a_cpu = np.arange(size, dtype=np.float32)
    b_cpu = np.arange(size, dtype=np.float32) * 2
    c_cpu = np.zeros(size, dtype=np.float32)

    # Allocate device memory
    a_gpu = cuda.to_device(a_cpu)
    b_gpu = cuda.to_device(b_cpu)
    c_gpu = cuda.device_array(size, dtype=np.float32)

    # Define block and grid dimensions
    threads_per_block = 256
    blocks_per_grid = (size + (threads_per_block - 1)) // threads_per_block

    # Launch kernels in different streams
    add_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu, stream=stream1)
    add_kernel[blocks_per_grid, threads_per_block](b_gpu, c_gpu, a_gpu, stream=stream2)

    # Wait for the streams to complete
    stream1.synchronize()
    stream2.synchronize()

    # Copy result back to host
    c_cpu = c_gpu.copy_to_host()

.. admonition:: Key Points
   :class: hint

    #. `@vectorize` can target GPUs.
    #. Device functions can only be invoked from another device functions or kernel functions.
    #. Data management is automatic in Numba, but we can also manage it manually.
    #. Streams can be used to run concurrent operations in GPUs.