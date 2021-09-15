print("First File Upload Test")

import matplotlib.pyplot as plt
import numpy as np
import scipy as stats

#1.4.5, Some Exercises

#1.4.5.1 Form the given array
#1
a = np.arange(1,16).reshape(3,5)
a = np.transpose(a)
b = np.array([a[1], a[3]])
#2.
a = np.arange(25).reshape(5,5)
b = np.array([1., 5, 10, 15, 20])
c = np.divide(a,b)
#3.
random = np.random.rand(30).reshape(10,3)
half = np.ones((10,3)) * 0.5
random_neg = np.abs(random - half)
sort_random = np.argsort(random_neg, axis=1)
random_index = sort_random[:,0]
#closest = random[random_index, random]

print(random[random_index])