# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:51:04 2024

@author: jcossc
"""

# algoritmo para sustitucion hacia delante
# n es el tamano de la dimension del problema
# matriz L, vector b ya estan dados como parametros
# guardar los resultados en el vector y
# Ly=b

def SustDelante(L,b):
  x=np.zeros_like(b)
  n=L.shape[0]# cantidad de renglones de L
  for i in range(n):
    sum=0.0
    for j in range(i):
      sum+=L[i,j]*x[j]
    x[i]=(b[i]-sum)/L[i,i]

  return x


# algoritmo para sustitucion hacia atras
# n es el tamano de la dimension del problema
# matriz U, vector y ya estan dados como parametros
# guardar los resultados en el vector x
# Ux=y
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
    

import numpy as np
from numpy import linalg as LA

U = np.array([[-4.,-3.,1.],[0.,5.,1.],[0.,0.,3.]])
L = np.array([[1,0,0],[-2,1,0],[-1,3,1]])

A=L@U

# Se definene los vectores
b = np.array([1.0,1.0,1.0])
# Usamos el algoritmo para encontrar la solucion
y = SustDelante(L, b)

yy = SustAtras(U, b)
# Se imprime el resultado
print(yy)

# # Usamos el algoritmo para encontrar la solucion
# x = sustAtras(U, y)

# # Se imprime el resultado
# print(x)

# print("comprobacion de los resultados")
# print(LA.solve(A,b))