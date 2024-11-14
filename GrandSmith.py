# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:47:38 2024

@author: jcossc
"""

import numpy as np
from numpy import linalg as la

A=[[1,-1,4.0],[1.0,4.0,-2.],[1,4.0,2.],[1,-1.,0.]]
A=np.array(A,dtype=float)

def GrandSmith(A):
  Q=np.empty_like(A) ## matriz Q
  R=np.zeros([A.shape[1],A.shape[1]]) ## matriz cuadrada
  vi=np.zeros([A.shape[1]])

  for i in range(A.shape[1]):
    vi=A[:,i]
    for j in range(i):
      R[j,i]=np.dot(Q[:,j].T,vi)
      vi = vi - R[j,i]*Q[:,j]
      #vi = a2 - (q1T, a2)* q1
    R[i,i]=np.linalg.norm(vi,2)
    Q[:,i]=vi/R[i,i]

  return Q,R

Q,R=GrandSmith(A)
# mostrar ambas matrices
print(Q)
print(R)

# comprobacion
print(np.matmul(Q,R))