# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:21:36 2025

@author: COAPA
"""

import numpy as np
from numpy import linalg as LA

def SustDelante(L,b):
  x=np.zeros_like(b)
  n=L.shape[0]# cantidad de renglones de L
  for i in range(n):
    sum=0.0
    for j in range(i):
      sum+=L[i,j]*x[j]
    x[i]=(b[i]-sum)/L[i,i]

  return x

def SustAtras(U,y):
    x=np.zeros_like(y)
    n=U.shape[0]# cantidad de renglones de U
    x[n-1] = y[n-1]/U[n-1,n-1]
    for i in range(n-2,-1,-1):
        sum=0.0
        for j in range(i+1,n):
            sum+=U[i,j]*x[j]
        x[i]=(y[i]-sum)/U[i,i]
    
    return x

def Cholesky(A):
    n=A.shape[0]
    L=np.zeros_like(A)

    for i in range(n):
      for j in range(i+1):
        if i==j:
          sum=0.0
          for k in range(i):
            sum+= L[i][k]*L[i][k]
          L[i][i]=np.sqrt(A[i][i]-sum)

        else:
          sum=0.0
          for k in range(j):
            sum+= L[i][k]*L[j][k]
          L[i][j]=(A[i][j]-sum)/L[j][j]



    return L

def SolveChol(A,b):
    L=Cholesky(A)
    y = SustDelante(L, b)
    x = SustAtras(L.T, y)
    
    return x

A = np.array([[6.0,15.0,55.0],[15.0,55.0,225.0],[55.0,225.0,979.0]])
b=np.array([1.,2.,3.])


sol=SolveChol(A,b)
print("Solucion")
print(sol)


y=LA.solve(A,b)
print("Solucion Analitica")
print(y)

