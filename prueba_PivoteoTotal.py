# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:42:06 2024

@author: jcossc
"""

import numpy as np

A=np.array([[1.0,2.0,4.0],[2.0,1.0,3.0],[3.0,2.0,4.0]])
b=np.array([1.,2.,3.])

print(A)

# Intercambiar las filas j = 0 y k = 2
A[[0, 2]] = A[[2, 0]]

# Intercambiar columnas j = 0 y k = 2
A[:, [0, 2]] = A[:, [2, 0]]

print(A)
