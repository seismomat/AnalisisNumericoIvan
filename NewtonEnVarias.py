# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:02:06 2024

@author: jcossc
"""

import numpy as np
import math
from numpy import linalg as LA

# funcion f1
def f1(x1, x2):
    return 2*x1-x2-math.e**(-x1)

# funcion f2
def f2(x1, x2):
    return -x1+2*x2-math.e**(-x2)

# parcial (en este caso la parcial de f1 y f2 es la misma)
def par(x):
    return 2+math.e**(-x)

def NewtonVarias(aprox):
    # contador de iteraciones
    n = 0
    while n < 100: # maximo numero de iteraciones 100
        '''es necesario calcular la matriz jacobiana
        y para ello se necesita una matriz vacia de 2x2'''
        jacob = np.zeros([2,2])

        # valores de la matriz jacobiana en todas sus entradas
        jacob[0][0] = par(aprox[0])
        jacob[0][1] = -1
        jacob[1][0] = -1
        jacob[1][1] = par(aprox[1])

        # guarda la evalucion de f1 y f2 en forma de vector
        fx = np.array(aprox)
        fx[0] = f1(aprox[0], aprox[1])
        fx[1] = f2(aprox[0], aprox[1])

        # FORMA ITERATIVA DEL METODO DE NEWTON PARA SISTEMAS NO LIENALES
        aprox = aprox - LA.solve(jacob, fx)

        # se incrementa el contador
        n+=1

    # El valor devuelto es la aproximacion
    return aprox



def main():
    #Aproximacion inicial (X=(0.0,0.0))
    ap = np.zeros([2])
    sol = NewtonVarias(ap)
    print('Aproximacion de la solucion')
    print(sol)
    print('Aproximacion evaluada en f1')
    print(f1(sol[0], sol[1]))
    print('Aproximacion evaluada en f2')
    print(f2(sol[0], sol[1]))

main()