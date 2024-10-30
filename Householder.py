# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:26:52 2024

@author: jcossc
"""
import numpy as np
from numpy import linalg as LA
A=np.array([[-1.,-1.,1.],[1.,3.,3.],[-1.,-1.,5.]])
#A=np.array([[1.,-1.,4.],[1.,4.,-2.],[1.,4.,2.],[1.,-1.,0.]])
#A=np.array([[-1.,-1.,1.],[1.,3.,3.],[-1.,-1.,5.],[1.,3.,7.]])
b=np.ones(A.shape[0])

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

def Householder(A):
  Q=np.eye(len(A))
  n=A.shape[1]

  for i in range(n):
    xi=np.zeros(len(A))
    xi[i:]=A[:,i][i:]
    norm_x=LA.norm(xi)
    ei=np.zeros(len(A))
    ei[i]=1.0
    ui=np.zeros(len(A))
    ui=xi+np.sign(A[i,i])*norm_x*ei
    vi=ui/LA.norm(ui)
    vi=vi.reshape(-1,1)

    H=np.eye(len(A))-2*vi@vi.T
    A=H@A
    A=np.round(A)
    Q=Q@H

  return Q,A

Q,R=Householder(A)
A_aux=Q@R
#y=Q.T@b
#x=SustAtras(R, y)