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

def TotalPivot(A,b):
    n = A.shape[0]
    U=np.copy(A)
    x=np.copy(b)
    P=np.eye(U.shape[0])
    Q=np.eye(U.shape[0])
    
    for k in range(n-1):
        max_row, max_col = np.unravel_index(np.argmax(np.abs(U[k:, k:])), (n - k, n - k))
        max_row += k
        max_col += k
        
        # intercambiar renglones
        U[[k,max_row]]=U[[max_row,k]]
        P[[k,max_row]]=P[[max_row,k]]
        x[[k,max_row]]=x[[max_row,k]]
        
        # intercambiar columnas
        U[:,[k,max_col]]=U[:,[max_col,k]]
        Q[:,[k,max_col]]=Q[:,[max_col,k]]


    return P,Q,U,x

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