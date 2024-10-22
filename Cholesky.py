# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:39:24 2024

@author: jcossc
"""
import numpy as np

A = np.array([[6.0,15.0,55.0],[15.0,55.0,225.0],[55.0,225.0,979.0]])

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

L=Cholesky(A)
print(L)