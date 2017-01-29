import numpy as np
import matplotlib.pyplot as plt 

def fun (n, x): 
    if x > 1:
        return 0.0
    elif 0 <= x <= 1:
        return 1 - x
    return 1 

vfun = np.vectorize(fun)

x = np.linspace(-10, 10, 100)    
y = vfun(3, x)

plt.plot(x, y, '-')
plt.show()