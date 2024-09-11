# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:05:40 2024

@author: jcossc
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg import lu

#Matriz con valores grandes
A = [[1E20 + 1, 1E20 + 2], [1E20 + 3, 1E20 + 4]];
A=np.array(A)
# Matriz con valores grandes pero más simples
B = [[1E20, 1E20],[1E20, 1E20]];  
B=np.array(B)

C = A - B;  # Resta de matrices

print(C)

eps=1E-9;
A = np.array([[1, 1-eps], [1-eps, 1]]);  # Matriz mal condicionada

# Realizar la descomposición LU
P, L, U = lu(A)

print('Matriz L:');
print(L);
print('Matriz U:');
print(U);
print('Matriz LU')
print(L@U)



