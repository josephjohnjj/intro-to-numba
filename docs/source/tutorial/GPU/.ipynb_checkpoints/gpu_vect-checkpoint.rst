Vectorization in GPU
--------------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 10 min

        **Objectives:**
            #. Learn how vectorize using GPUs.


The code below Python function adds a given scalar (`y`) to all elements of a one-dimensional array. 
The more interesting aspects lie in the function's declaration, which includes two key elements:

..  code-block:: python
    :linenos:

    from numba import guvectorize, int64
    import numpy as np
    
    @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
    def g(x, y, res):
        for i in range(x.shape[0]):
            res[i] = x[i] + y

*(n),()->(n)* tells NumPy that the function takes a n-element one-dimension array, a scalar, denoted 
by the empty *tuple ()*, and computes an *n-element one-dimension array*.

Unlike *vectorize()* functions, *guvectorize()* functions should not return any result.

In Numba's `@guvectorize` functions, there is no explicit return statement. Instead, the output is passed via the output argument (in this case, `res`). Numba modifies this array in place. When invoking the function, the result is automatically returned because Numba allocates an output array for you.


**Key Points:**

1. **In-place Modification**: The `res` array is the output, which is modified in place within the `guvectorize` function.
2. **Return**: When calling the `guvectorize`-decorated function, even though the function doesn't explicitly return anything, Numba provides the output array based on the function signature.



**How the Return Works:**

- **`result = g(x, y)`**: Numba handles the allocation of the `res` array internally and returns it automatically after the function finishes.
- You don't need to declare or pre-allocate the `res` array when calling the function; Numba will do this for you.
  
Thus, the array `result` contains the values produced by the in-place modification of `res` inside the `g` function.


.. admonition:: Key Points
   :class: hint

    #. `@guvectorize` can be used to vectorize the function in GPU. 