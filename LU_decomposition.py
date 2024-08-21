# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:43:57 2024

@author: jcossc
"""
import numpy as np
from numpy import linalg as LA
from scipy.linalg import lu
A=np.array([[1,2,3],[2,3,1],[-2,3,-2]])


def LU(A):
  U=np.copy(A)
  L=np.eye(A.shape[0])
  L1=np.eye(A.shape[0])
  for k in range(A.shape[0]):
    L=np.eye(A.shape[0])
    for i in range(k+1,A.shape[0]):
      L[i,k]=-U[i,k]/U[k,k]
    U=L@U
    L1=L1@L
    #L1=2*np.eye(A.shape[0])-L1
  return L1,U

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
#L,U=factorizacionLU(A)
print("Matriz L")
print(L)
print("Matriz U")
print(U)

print("comprobacion")
print("Matriz LU")
print(L@U)
print("Matriz A")
print(A)
"""# Descomposición LU
P, L, U = lu(A)

print("Matriz de permutación P:")
print(P)
print("Matriz triangular inferior L:")
print(L)
print("Matriz triangular superior U:")
print(U)"""