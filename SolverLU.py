# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:19:33 2024

@author: jcossc
"""

import numpy as np
from numpy import linalg as LA

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

def SustDelante(L,b):
    """
    Realiza la sustitución hacia adelante para resolver un sistema de ecuaciones
    lineales de la forma Lx = b, donde L es una matriz triangular inferior.
    
    Parámetros:
    -----------
    L : numpy.ndarray
        Matriz triangular inferior de tamaño n x n.
    b : numpy.ndarray
        Vector de términos independientes de tamaño n.
    
    Retorna:
    --------
    x : numpy.ndarray
        Vector solución de tamaño n, tal que Lx = b.
    
    Descripción:
    ------------
    La función aplica el método de sustitución hacia adelante, en el cual se resuelve
    una matriz triangular inferior. Este método es adecuado cuando se conoce una factorización LU
    de la matriz, y L es la matriz triangular inferior resultante. El algoritmo itera
    sobre los renglones de L para calcular las componentes de x en orden ascendente.
    
    Ejemplo de uso:
    ---------------
    L = np.array([[2, 0, 0],
                  [3, 1, 0],
                  [1, -2, 1]])
    b = np.array([4, 5, 6])
    x = SustDelante(L, b)
    """

    x=np.zeros_like(b)
    n=L.shape[0]# cantidad de renglones de L
    for i in range(n):
        sum=0.0
        for j in range(i):
            sum+=L[i,j]*x[j]
        x[i]=(b[i]-sum)/L[i,i]
    
    return x


def SustAtras(U, y):
    """
    Realiza la sustitución hacia atrás para resolver un sistema de ecuaciones
    lineales de la forma Ux = y, donde U es una matriz triangular superior.

    Parámetros:
    -----------
    U : numpy.ndarray
        Matriz triangular superior de tamaño n x n.
    y : numpy.ndarray
        Vector de términos independientes de tamaño n.

    Retorna:
    --------
    x : numpy.ndarray
        Vector solución de tamaño n, tal que Ux = y.

    Descripción:
    ------------
    La función aplica el método de sustitución hacia atrás, que es útil para resolver
    sistemas de ecuaciones lineales cuando se tiene una matriz triangular superior. Este método
    comienza resolviendo la última ecuación y continúa hacia arriba para calcular las componentes
    de x en orden descendente. Es comúnmente utilizado después de la factorización LU.

    Ejemplo de uso:
    ---------------
    U = np.array([[3, 2, 1],
                  [0, 4, 5],
                  [0, 0, 6]])
    y = np.array([5, 7, 8])
    x = SustAtras(U, y)
    """
    x = np.zeros_like(y)
    n = U.shape[0]  # cantidad de renglones de U
    x[n-1] = y[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum = 0.0
        for j in range(i+1, n):
            sum += U[i, j] * x[j]
        x[i] = (y[i] - sum) / U[i, i]
    
    return x


def Solve(A, b):
    """
    Resuelve un sistema de ecuaciones lineales de la forma Ax = b utilizando
    la factorización LU y los métodos de sustitución hacia adelante y hacia atrás.

    Parámetros:
    -----------
    A : numpy.ndarray
        Matriz cuadrada de tamaño n x n que representa el sistema de ecuaciones.
    b : numpy.ndarray
        Vector de términos independientes de tamaño n.

    Retorna:
    --------
    x : numpy.ndarray
        Vector solución de tamaño n, tal que Ax = b.

    Descripción:
    ------------
    La función resuelve el sistema lineal Ax = b mediante los siguientes pasos:
    1. Se realiza la factorización LU de la matriz A, donde A = LU.
       - L es una matriz triangular inferior.
       - U es una matriz triangular superior.
    2. Se resuelve el sistema Ly = b mediante sustitución hacia adelante para obtener y.
    3. Se resuelve el sistema Ux = y mediante sustitución hacia atrás para obtener x.

    Esta técnica es especialmente útil para sistemas grandes, ya que la factorización LU permite 
    resolver el sistema de manera eficiente.

    Ejemplo de uso:
    ---------------
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [3, 3, 5]])
    b = np.array([2, -1, 3])
    x = Solve(A, b)
    """
    L, U = LU(A)
    y = SustDelante(L, b)
    x = SustAtras(U, y)
    
    return x

    
    
# b = np.array([1.0,1.0,1.0])
# A=np.array([[2,1,1],[4,-6,0],[-2,7,2]])
# SOl=Solve(A,b)

# print("Solución de Ax = b")
# print(SOl)

# print("Solución de Ax = b con numpy")
# print(LA.solve(A,b))

A=np.array([[4,12,-16],[12,37,-43],[-16,-43,98]])
L, U = LU(A)