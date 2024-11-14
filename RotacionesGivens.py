# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:47:38 2024

@author: jcossc
"""

import numpy as np
from numpy import linalg as la

A=[[1,-1,4.0],[1.0,4.0,-2.],[1,4.0,2.],[1,-1.,0.]]
A=np.array(A,dtype=float)

def Givens(A):
  R=np.copy(A)
  Q=np.eye(len(A))
  for i in range(A.shape[1]): # maneja el pivote
    for j in range(i+1,len(A)):
      G=np.eye(len(R))
      ro=la.norm([R[i,i],R[j,i]])
      c=R[i,i]/ro
      s=R[j,i]/ro
      G[i,i]=c; G[i,j]=s;
      G[j,j]=c; G[j,i]=-s;
      R=G@R

      R=np.round(R,5)
      G=np.round(G,5)
      Q=Q@G.T
  return Q,R

Q,R=Givens(A)
# mostrar ambas matrices
print(Q)
print(R)

# comprobacion
print(np.matmul(Q,R))