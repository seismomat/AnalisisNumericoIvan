# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:40:40 2025

@author: jcossc
"""
import numpy as np
from numpy import linalg as LA
from LU_decomposition import LU as LUU
from SolverLU import Solve

A=np.array([[2.0,3.0,4.0],[4.0,7.0,5.0],[4.0,9.0,5.0]])
b=np.array([1.,2.,3.])

def TotalPivot(A, b):
    n = A.shape[0]  # Tamaño de la matriz (n x n)
    U = np.copy(A)  # Copia de la matriz A para no modificarla directamente
    x = np.copy(b)  # Copia del vector b
    P = np.eye(n)   # Matriz de permutación de filas (inicialmente identidad)
    Q = np.eye(n)   # Matriz de permutación de columnas (inicialmente identidad)

    for k in range(n - 1):
        # --- PASO 1: Encontrar el pivote máximo 
        # en la submatriz U[k:, k:] ---
        # np.abs(U[k:, k:]): Valores absolutos de la 
        # submatriz desde (k,k) hasta el final
        # np.argmax(...): Índice lineal del elemento 
        # con mayor valor absoluto (en arreglo aplanado)
        # np.unravel_index(...): Convierte el índice lineal 
        # a coordenadas (fila, columna) en la submatriz
        max_row, max_col = np.unravel_index(np.argmax(np.abs(U[k:, k:])), (n - k, n - k))
        
        # Ajustar índices para referenciar la posición en la matriz original U (no solo la submatriz)
        max_row += k
        max_col += k

        # --- PASO 2: Intercambiar filas para llevar el pivote a la posición (k,k) ---
        # Intercambia filas en U
        U[[k, max_row]] = U[[max_row, k]]
        # Intercambia filas en la matriz de permutación P
        P[[k, max_row]] = P[[max_row, k]]
        # Intercambia elementos en el vector x
        x[[k, max_row]] = x[[max_row, k]]

        # --- PASO 3: Intercambiar columnas para optimizar la factorización ---
        # Intercambia columnas en U
        U[:, [k, max_col]] = U[:, [max_col, k]]
        # Intercambia columnas en la matriz de permutación Q
        Q[:, [k, max_col]] = Q[:, [max_col, k]]

    return P, Q, U, x  # Devuelve: Matrices de permutación, matriz triangular, y vector modificado

P,Q,U,x=TotalPivot(A,b)

#bb=P.T@U@Q.T
#U[[j,k]]=U[[k,j]]
#U[:,[j,k]]=U[:,[k,j]]

def Solver_LU_Pivot_Total(A,b):
    P,Q,A_g,b_g=TotalPivot(A,b)
    x=Solve(A_g,b_g)
    x=Q@x
    
    return x

sol=Solver_LU_Pivot_Total(A,b)
print("Solucion")
print(sol)


y=LA.solve(A,b)
print("Solucion Analitica")
print(y)