# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:09:41 2024

@author: jcossc
"""

import numpy as np

A=np.array([[1.0,2.0,4.0],[2.0,1.0,3.0],[3.0,2.0,4.0]])

def LU_PartialPivot(A):
  U=np.copy(A)
  Ps=[]
  Ls=[]
  for j in range(U.shape[0]):
    P=np.eye(U.shape[0])
    L=np.eye(U.shape[0])
    k=np.argmax(np.abs(U[j:,j]))+j
    print(f"k={k}")
    U[[j,k]]=U[[k,j]]
    P[[j,k]]=P[[k,j]]
    for i in range(j+1,A.shape[0]):
      L[i,j]=-U[i,j]/U[j,j]
    U=L@U

    Ps.append(P)
    Ls.append(np.linalg.inv(L))

  PM=np.eye(U.shape[0])
  for i in range(len(Ps)-1,-1,-1):
    PM=(Ps[i]@Ls[i])@PM
  PM

  return PM,U

L,U=LU_PartialPivot(A)

print("Matriz L")
print(L)
print("Matriz U")
print(U)

print("comprobacion")
print("Matriz LU")
print(L@U)
print("Matriz A")
print(A)