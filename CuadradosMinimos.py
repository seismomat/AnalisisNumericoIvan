# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:48:19 2024

@author: jcossc
"""

from mpl_toolkits import mplot3d
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

f=lambda x: 2*x+ 1

def gen_data(n):
    x = np.linspace(0,1,n)
    y = np.zeros_like(x)
    eps=1E-1
    for i in range(n):
        y[i] = f(x[i]+eps**random.randint(0,10))

    return x, y

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

def SustDelante(L,b):
    x=np.zeros_like(b)
    n=L.shape[0]# cantidad de renglones de L
    for i in range(n):
        sum=0.0
        for j in range(i):
            sum+=L[i,j]*x[j]
        x[i]=(b[i]-sum)/L[i,i]
    
    return x


def SustAtras(U, y):
    x = np.zeros_like(y)
    n = U.shape[0]  # cantidad de renglones de U
    x[n-1] = y[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum = 0.0
        for j in range(i+1, n):
            sum += U[i, j] * x[j]
        x[i] = (y[i] - sum) / U[i, i]
    
    return x

def EcNormal():
    x,y = gen_data(20)
    A=np.zeros((len(y),2))
    A[:,0]=1.0
    A[:,1]=x
    b=y
    
    AtA=A.T@A
    Atb=A.T@b
    
    L=Cholesky(AtA)
    Lt=L.T
    
    ySol = SustDelante(L, Atb)
    Params = SustAtras(Lt, ySol)
    z=Params[0]+Params[1]*x
    
    plt.plot(x,y,'ro')
    plt.plot(x,z,'black')
    plt.show()
    
    return x

x=EcNormal()



