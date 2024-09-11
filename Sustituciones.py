# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:51:04 2024

@author: jcossc
"""

# algoritmo para sustitucion hacia delante
# n es el tamano de la dimension del problema
# matriz L, vector b ya estan dados como parametros
# guardar los resultados en el vector y
# Ly=b
def sustDelante(L, b):
    n=len(L)
    y=np.empty_like(b)
    y[0] = b[0]
    for i in range(1,n):
        y[i] = b[i]
        for j in range(0,i):
            y[i] -= L[i][j]*y[j]
    return y

# algoritmo para sustitucion hacia atras
# n es el tamano de la dimension del problema
# matriz U, vector y ya estan dados como parametros
# guardar los resultados en el vector x
# Ux=y
def sustAtras(U, y):
    n=len(U)
    x=np.empty_like(y)
    x[n-1] = y[n-1]/U[n-1][n-1]
    for i in range(n-2,-1,-1):
        x[i] = y[i]
        for j in range(i+1,n):
            x[i] -= U[i][j]*x[j]
        x[i] /= U[i][i]
    return x

import numpy as np
from numpy import linalg as LA

U = np.array([[-4,-3,1],[0,5,1],[0,0,3]])
L = np.array([[1,0,0],[-2,1,0],[-1,3,1]])

A=L@U

# Se definene los vectores
b = np.array([5.0,6.0,1.0])
# Usamos el algoritmo para encontrar la solucion
y = sustDelante(L, b)

# Se imprime el resultado
print(y)

# Usamos el algoritmo para encontrar la solucion
x = sustAtras(U, y)

# Se imprime el resultado
print(x)

print("comprobacion de los resultados")
print(LA.solve(A,b))