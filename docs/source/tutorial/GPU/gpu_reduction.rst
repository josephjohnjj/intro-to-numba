Reduction in GPU
--------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 10 min

        **Objectives:**
            #. Learn how to perform reduction operations on GPUs.


Numba offers a `@reduce` decorator that transforms a simple binary operation into a reduction kernel.

..  code-block:: python
    :linenos:

    import numpy
    from numba import cuda

    @cuda.reduce
    def sum_reduce(a, b):
        return a + b

    A = (numpy.arange(1234, dtype=numpy.float64)) + 1
    expect = A.sum()      # NumPy sum reduction
    got = sum_reduce(A)   # cuda sum reduction
    assert expect == got

*(n),()->(n)* tells NumPy that the function takes a n-element one-dimension array, a scalar, denoted 
by the empty *tuple ()*, and computes an *n-element one-dimension array*.

Unlike *vectorize()* functions, *guvectorize()* functions should not return any result.

.. admonition:: Key Points
   :class: hint

    #. `@reduce` can convert a simple binary operation into a reduction kernel.