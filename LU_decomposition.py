# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:43:57 2024

@author: jcossc
"""
import numpy as np
from numpy import linalg as LA
from scipy.linalg import lu
A=np.array([[1,2,3],[2,3,1],[-2,3,-2]])

A=np.array([[2,1,1],[4,-6,0],[-2,7,2]])
import numpy as np

import numpy as np

def LU(A):
    """
    Realiza la descomposición LU de una matriz cuadrada A.

    La descomposición LU descompone una matriz A en el producto de una matriz triangular inferior L y una matriz triangular superior U, tales que A = L * U.

    Parámetros:
    ----------
    A : numpy.ndarray
        Matriz cuadrada de tamaño n x n.

    Retorna:
    -------
    L : numpy.ndarray
        Matriz triangular inferior L de tamaño n x n, con 1s en la diagonal principal.
    U : numpy.ndarray
        Matriz triangular superior U de tamaño n x n.

    Ejemplo:
    --------
    >>> A = np.array([[4, 3], [6, 3]])
    >>> L, U = LU(A)
    >>> print("L:", L)
    >>> print("U:", U)
    L: [[ 1.   0. ]
        [ 1.5  1. ]]
    U: [[4.  3. ]
        [0.  -1.5]]

    Notas:
    ------
    - Esta implementación no incluye pivoteo, por lo que es necesario que A no tenga ceros en su diagonal principal para evitar divisiones por cero.
    - El algoritmo modifica la matriz U en cada iteración para ir obteniendo la matriz triangular superior.
    - La matriz L se construye como el producto acumulado de las matrices de eliminación en cada paso.

    """

    U = np.copy(A)
    L = np.eye(A.shape[0])
    L1 = np.eye(A.shape[0])

    for k in range(A.shape[0]):
        L = np.eye(A.shape[0])
        for i in range(k + 1, A.shape[0]):
            L[i, k] = -U[i, k] / U[k, k]
        U = L @ U
        L1 = L1 @ L

    L1 = 2 * np.eye(A.shape[0]) - L1
    return L1, U



def factorizacionLU(A):
    # dimension de la matriz
    n = len(A)
    # matriz L inicialmente es la identidad
    L = np.identity(n)
    # inicialmente la matriz A y la matriz U son iguales
    U = np.zeros((n,n))
    U=np.copy(A)
    # eliminacion gaussiana
    for i in range(n):
        for j in range(i+1,n):
            # guardar los factores de eliminacion gaussiana 
            # en la matriz L
            factor = U[j][i]/U[i][i]
            L[j][i] = factor
            # realizar eliminacion gaussiana en la matriz U
            # para quedar de forma triangular superior
            for k in range(i,n):
                U[j][k] = U[j][k] - factor*U[i][k]
    return L,U
L,U=LU(A)
L,U=factorizacionLU(A)
print("Matriz L")
print(L)
print("Matriz U")
print(U)

# print("comprobacion")
# print("Matriz LU")
# print(L@U)
# print("Matriz A")
# print(A)


"""# Descomposición LU
P, L, U = lu(A)

print("Matriz de permutación P:")
print(P)
print("Matriz triangular inferior L:")
print(L)
print("Matriz triangular superior U:")
print(U)"""