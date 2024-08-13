Automatic Parallelisation
--------------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 10 min

        **Objectives:**
            #. Learn how to automatically parallelize code in Numba.

Setting the `parallel` option for `jit()` enables Numba to automatically parallelize and optimize 
parts of a function, though it currently only works on CPUs. Instead of parallelizing each operation 
individually, which can lead to inefficiencies, Numba identifies and fuses adjacent parallelizable 
operations into kernels that run in parallel, improving performance.


..  code-block:: python
    :emphasize-lines: 1
    :linenos:

    @jit(nopython=True, parallel=True)
    def reduction_with_parallel(n):
        shp = (13, 17)
        result1 = 2 * np.ones(shp, np.int_)
        tmp = 2 * np.ones_like(result1)

        for i in prange(n):
            result1 *= tmp

        return result1


.. admonition:: Key Points
   :class: hint

    #. `@jit(nopython=True, parallel=True)` automatically parallelise functions.
    #. This functionality only works for CPU.