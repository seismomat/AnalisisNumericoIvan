# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:21:00 2024

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
    P=np.eye(U.shape[0])
    for j in range(U.shape[0]):
        #P=np.eye(U.shape[0])
        k=np.argmax(np.abs(U[j:,j]))+j
        U[[j,k]]=U[[k,j]]
        P[[j,k]]=P[[k,j]]
        x[[j,k]]=x[[k,j]]


    return P,U,x

def Solver_LU_Pivot_Partial(A,b):
    Ps,A_g,b_g=PartialPivot(A,b)
    x=Solve(A_g,b_g)
    
    return x

sol=Solver_LU_Pivot_Partial(A,b)
print("Solucion")
print(sol)

from scipy.linalg import lu
P1, L1, U1 = lu(A)

y=LA.solve(A,b)
print("Solucion Analitica")
print(y)