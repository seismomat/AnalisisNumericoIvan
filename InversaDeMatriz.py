# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:38:29 2025

@author: jcossc
"""

import numpy as np
from numpy import linalg as LA

A=np.array([[2,3,-4],[0,-4,2],[1,-1,5]])


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

def CoFac(Mat):
  M_CoFa=np.zeros_like(Mat)

  for i in range(Mat.shape[0]):# va sobre renglones
    for j in range(Mat.shape[1]):#va sobre columnas
      M_CoFa[i][j]=((-1)**(i+j))*Det(SubMat(Mat,i,j))

  return M_CoFa

def Inv(M):
  deter=Det(M)
  M_CoFa=CoFac(M)
  return (1/deter)*Transpuesta(M_CoFa)

A_inv=Inv(A)

Id=np.round(A@A_inv,2)
print(Id)