import numpy as np
a = np.array([2, 4, 6, 8, 9 , 30])
z = np.argwhere(a <5).flatten()
print(z)