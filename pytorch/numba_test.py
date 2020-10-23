import numpy as np
from numba import vectorize

# 
@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
    return a + b

n = 10000000
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=a.dtype)
c = np.empty_like(a, dtype=a.dtype)

# Add Arrays on GPU
c = Add(a, b)
print (c)
