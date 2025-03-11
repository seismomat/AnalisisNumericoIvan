# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:57:40 2025

@author: jcossc
"""

import numpy as np
from numpy import linalg as LA

def SubMat(Mat,ren,col):
  M1=np.copy(Mat)
  M1=np.delete(M1,ren,axis=0)# elimina sobre sobre los renglones
  M1=np.delete(M1,col,axis=1)# elimina sobre sobre los renglones
  return M1

def Det(Mat):
  if Mat.shape[0]==2 and Mat.shape[1]==2:
    return Mat[0][0]*Mat[1][1]-(Mat[0][1]*Mat[1][0])
  else:
    deter=0.0
    for col in range(Mat.shape[0]):
      deter+= ((-1)**col)*Mat[0][col]*Det(SubMat(Mat,0,col))
    return deter

def Transpuesta(Mat):
    for ren in range(Mat.shape[0]):
        for col in range(Mat.shape[1]):
            if ren<col:
                Mat[ren,col],Mat[col,ren]=Mat[col,ren],Mat[ren,col]
            
    return Mat

def Cofactores(Mat):
  Cofa=np.zeros_like(Mat)

  for ren in range(Mat.shape[0]):
    for col in range(Mat.shape[1]):
      Cofa[ren,col]=((-1)**(ren+col))*Det(SubMat(Mat,ren,col))

  return Cofa

def Inv(Mat):
  deter=Det(Mat)
  Cofac=Cofactores(Mat)
  Cofac=Transpuesta(Cofac)

  Inversa=(1/deter)*Cofac

  return Inversa

A=np.array([[2,3,-4],[0,-4,2],[1,-1,5]])
b=np.array([1.0,1.0,1.0])

def SolveInv(Mat,vec):
  InvMat=Inv(Mat)
  Solucion=InvMat@vec

  return Solucion

Sol=SolveInv(A,b)
print(Sol)
Sol_python=LA.solve(A,b)
print(Sol_python)
"""
print(A)
InvA=Inv(A)
print(A@InvA)
print(InvA@A)
"""