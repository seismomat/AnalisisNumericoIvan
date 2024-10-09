# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:33:37 2024

@author: jcossc
"""
import numpy as np
from numpy import linalg as LA
from LU_decomposition import LU as LUU
from SolverLU import Solve 

A=np.array([[1.0,2.0,4.0],[2.0,1.0,3.0],[3.0,2.0,4.0]])
#A=np.array([[2.0,1.0,1.0],[4.0,-6.0,0.0],[-2.0,7.0,2.0]])
b=np.array([1.,2.,3.])

def LU_PartialPivot(A):
    """
    Realiza la descomposición LU con pivoteo parcial de una matriz 
    cuadrada A.

    La descomposición LU con pivoteo parcial descompone una matriz A 
    en el producto de una matriz de permutación P, una matriz triangular 
    inferior L, y una matriz triangular superior U, tales que 
    P * A = L * U.

    Parámetros:
    ----------
    A : numpy.ndarray
        Matriz cuadrada de tamaño n x n.

    Retorna:
    -------
    PM : numpy.ndarray
        Matriz de permutación y transformación final que combina las 
        permutaciones y eliminaciones.
    U : numpy.ndarray
        Matriz triangular superior U de tamaño n x n.

    Ejemplo:
    --------
    >>> A = np.array([[4, 3], [6, 3]])
    >>> PM, U = LU_PartialPivot(A)
    >>> print("PM:", PM)
    >>> print("U:", U)
    PM: [[0. 1.]
         [1. 0.]]
    U: [[6.  3. ]
        [0.  1.5]]

    Notas:
    ------
    - La función realiza pivoteo parcial, lo que significa que se 
    reordena la matriz A para evitar errores numéricos debidos a elementos pequeños en la diagonal.
    - En cada iteración, se selecciona el mayor valor absoluto en 
    la columna actual para el pivote.
    - Las matrices de permutación (P) y eliminación (L) se acumulan 
    en listas durante la ejecución y se combinan al final para formar la matriz de permutación y transformación final PM.
    - Esta implementación no devuelve explícitamente la matriz L, 
    pero L puede ser reconstruida utilizando PM y U si es necesario.

    """
    U=np.copy(A)
    Ps=[]
    Ls=[]
    
    L1 = np.eye(U.shape[0])
    for j in range(U.shape[0]):
        P=np.eye(U.shape[0])
        L=np.eye(U.shape[0])
        k=np.argmax(np.abs(U[j:,j]))+j
        #print(f"k={k}")
        U[[j,k]]=U[[k,j]]
        P[[j,k]]=P[[k,j]]
        
        for i in range(j+1,A.shape[0]):
          L[i,j]=-U[i,j]/U[j,j]
        U=L@U
        
        L1 = L1 @ L
    
        Ps.append(P)
        Ls.append(np.linalg.inv(L))

    L1 = 2 * np.eye(A.shape[0]) - L1
    PM=np.eye(U.shape[0])
    for i in range(len(Ps)-1,-1,-1):
        PM=(Ps[i]@Ls[i])@PM

    return PM,U,L1

L,U,LL=LU_PartialPivot(A)

from scipy.linalg import lu
P1, L1, U1 = lu(A)

# y=LA.solve(L1,b_g)
# solAna=LA.solve(U1,y)
# print("Solucion Analitica")
# print(solAna)