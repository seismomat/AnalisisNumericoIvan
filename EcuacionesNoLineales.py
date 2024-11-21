# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:22:35 2024

@author: jcossc
"""

import numpy as np

def Biseccion(Tol,N,f,a,b):
  #N es el numero de iteraciones
  # Tol es la tolerancia
  # f es la función a la cual se le quiere
  # obtener las raíces
  # a es el extremo izquierdo del intervalo
  # b es el extremo derecho del intervalo

  fa,fb=f(a),f(b)
  if fa*fb>0.0:
    print("No tiene raíces en el intervalo")

  x0=0.0
  Iter=0
  while Iter<=N:
    x1=(a+b)/2.0
    fx1=f(x1)
    if abs(fx1)<=Tol and abs(x1-x0)<=Tol:
      print("Tu raiz es "+str(x1))
      return x1

    if fa*fx1<0:
      b=x1
    if fx1*fb<0:
      a=x1

    x0=x1
    Iter+=1

  else:
    print("El valor aproximado de tu raiz es "+str(x1))
    
    
    
def ReglaFalsa(Tol,N,f,a,b):
  #N es el numero de iteraciones
  # Tol es la tolerancia
  # f es la función a la cual se le quiere
  # obtener las raíces
  # a es el extremo izquierdo del intervalo
  # b es el extremo derecho del intervalo

  fa,fb=f(a),f(b)
  if fa*fb>0.0:
    print("No tiene raíces en el intervalo")

  x0=0.0
  Iter=0
  while Iter<=N:
    x1=(a*fb-b*fa)/(fb-fa)
    fx1=f(x1)
    if abs(fx1)<=Tol and abs(x1-x0)<=Tol:
      print("Tu raiz es "+str(x1))
      return x1

    if fa*fx1<0:
      b=x1
    if fx1*fb<0:
      a=x1

    x0=x1
    Iter+=1

  else:
    print("El valor aproximado de tu raiz es "+str(x1))
    
    
def Secante(f, Tol, N, x0, x1):
    #N es el numero de iteraciones
    # Tol es la tolerancia
    # f es la función a la cual se le quiere
    # obtener las raíces
    # a es el extremo izquierdo del intervalo
    # b es el extremo derecho del intervalo
    
    #contador de iteraciones
    n = 1
    #mientras no se haya superado el limite de iteraciones
    while( N >= n ):
        #calculo de los valores de fx
        fx0,fx1=f(x0),f(x1)
        #se calcula la siguiente aproximacion
        xn = x1-fx1*((x1-x0)/float(fx1-fx0))
        #en caso de que se cumplan los criteros de paro
        #se devuelve la raiz
        if (abs(f(xn)) <= Tol and abs(x0-x1) <= Tol):
            return xn
        #se actualizan los valores
        #print(x0,x1)
        x0 = x1
        x1 = xn
        
        #incremento en las iteraciones
        n+=1

    else:
        print("El valor aproximado de tu raiz es "+str(x1))

def Secante1(f,Tol,N,a,b):
  #N es el numero de iteraciones
  # Tol es la tolerancia
  # f es la función a la cual se le quiere
  # obtener las raíces
  # a es el extremo izquierdo del intervalo
  # b es el extremo derecho del intervalo

    fa,fb=f(a),f(b)
    if fa*fb>0.0:
        print("No tiene raíces en el intervalo")

    Iter=0
    x0=a
    while Iter<=N:
        fa,fb=f(a),f(b)
        x1=b-(fb*(b-a)/(fb-fa))
        fx1=f(x1)
        if abs(fx1)<=Tol and abs(x1-x0)<=Tol:
            print("Tu raiz es "+str(x1))
            return x1

        if fa*fx1<0:
            b=x1
        if fx1*fb<0:
            a=x1
        x0=x1

        Iter+=1

    else:
        print("El valor aproximado de tu raiz es"+str(x1))
def Newton(f,df,Tol,N,x0):
  Iter=0

  while Iter<=N:
    fx=f(x0)
    dfx=df(x0)

    xn = x0 - (fx/float(dfx))
    if abs(f(xn))<=Tol and abs(xn-x0)<=Tol:
      print("Tu raiz es "+str(xn))
      return xn

    x0=xn

    Iter+=1

  else:
    print("El valor aproximado de tu raiz es "+str(xn))
    
def FixPoint(g,x0,Tol=0.0001,N=100):
    n=1
    while n<=N:
        x1=g(x0)
        if abs(x1-x0)<=Tol:
            print(f"EL punto fijo es {x1}")
            return x1
        x0=x1
        n+=1
    else:
        print(f"Una aproximación del punto fijo es {x1}")
        return x1
    
f=lambda x:(x**2)-1
df= lambda x:2*x

Biseccion(1E-4,100,f,0.5,1.5)
ReglaFalsa(1E-4,100,f,0.5,1.5)
Secante(f,0.001,20,0.5,1.5)
Secante1(f,0.001,20,0.5,1.5)
Newton(f,df,1E-5,100,1.)

f= lambda x: np.cos(x)-x+1
g=lambda x: np.cos(x)+1
# condicion inicial
x=1.5

FixPoint(g,1.5)