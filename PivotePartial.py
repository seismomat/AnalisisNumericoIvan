# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:56:22 2024

@author: jcossc
"""

import numpy as np
from numpy import linalg as LA
from LU_decomposition import LU as LUU
from SolverLU import Solve 

A=np.array([[1.0,2.0,4.0],[2.0,1.0,3.0],[3.0,2.0,4.0]])
A=np.array([[2.0,1.0,1.0],[4.0,-6.0,0.0],[-2.0,7.0,2.0]])
b=np.array([1.,2.,3.])

def PartialPivot(A,b):
    U=np.copy(A)
    x=np.copy(b)
    Ps=[]
    for j in range(U.shape[0]):
        P=np.eye(U.shape[0])
        k=np.argmax(np.abs(U[j:,j]))+j
        U[[j,k]]=U[[k,j]]
        P[[j,k]]=P[[k,j]]
        b[[j,k]]=b[[k,j]]
   
        Ps.append(P)

    return Ps,U,b

from scipy.linalg import lu
P1, L1, U1 = lu(A)
print("Matriz L1")
print(L1)
print("Matriz U1")
print(U1)
print("Matriz P1")
print(P1)

Ps,A_g,b_g=PartialPivot(A,b)
L,U=LUU(A_g)
print("Matriz L")
print(L)
print("Matriz U")
print(U)
print("Matriz A")
print(L@U)
x=Solve(A_g,b_g)
print("A gorro")
print(A_g)
print("b gorro")
print(b_g)
print("Solucion")
print(x)
print("Solucion numpy")
print(LA.solve(A_g,b_g))
print("Solucion original")
print(LA.solve(A,b))