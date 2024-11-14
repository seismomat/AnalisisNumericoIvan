# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:47:38 2024

@author: jcossc
"""
import numpy as np

def trid(d1, d0,  d2, b):
    #d1 diagonal superior
    #d0 diagonal principal
    #d2 diagonal inferior
    n=d0.shape[0]
    x=np.zeros(n)
    for i in range(n-1):
        a=0.0
        a=d1[i]/d0[i]
        d0[i+1]=d0[i+1]-a*d2[i]
        b[i+1]=b[i+1]-a*b[i]

    x[n-1]=b[n-1]/d0[n-1]
    for i in range(n-2,-1,-1):
        x[i]=(b[i]-d2[i]*x[i+1])/d0[i]
    return x



n=5
dprin=-2.0*np.ones(n)
dinf=1.0*np.ones(n-1)
dsup=1.0*np.ones(n-1)

b=1.0*np.ones(n)

res=trid(dsup,dprin,dinf,b)
res

Mprin=np.diag(np.ones(n))*(-2.0)
Msup=np.diag(np.ones(n-1),1)*1.0
Minf=np.diag(np.ones(n-1),-1)*1.0
M=Mprin+Minf+Msup

res1=np.linalg.solve(M,b)
res1