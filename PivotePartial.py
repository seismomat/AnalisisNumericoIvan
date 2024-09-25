# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:56:22 2024

@author: jcossc
"""

import numpy as np
from LU_decomposition import LU

A=np.array([[1.0,2.0,4.0],[2.0,1.0,3.0],[3.0,2.0,4.0]])
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

Ps,A_g,b_g=PartialPivot(A,b)
L,U=LU(A_g)
print("Matriz A gorro")
print(A_g)
print("Matriz b gorro")
print(b_g)
print("Matriz L")
print(L)
print("Matriz U")
print(U)

print("comprobacion")
print("Matriz LU")
print(L@U)