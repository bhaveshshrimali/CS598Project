# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 00:55:45 2017

@author: M K Shrimali
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as lge
import numpy.polynomial.chebyshev as che
from scipy.optimize import fsolve,root
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_bvp


Ns=80 #80
fB=2.
fA=1.
Mu=1.
Kappa=20

class QuadratureInfo:
    def __init__(self, npoints):
        self.npoints=npoints
        A=self.A=1.e-3
        B=self.B=2.e-3
        self.ref_nodes,self.ref_weights=lge.leggauss(npoints)
        self.curve_nodes=0.5*((B-A)*self.ref_nodes+(B+A))
        self.curve_weights=(B-A)*self.ref_weights

Qs=QuadratureInfo(Ns)
xi,omga,A,B=Qs.curve_nodes,Qs.curve_weights,Qs.A,Qs.B
J=np.arange(1,xi.size-1)

def Fp(f):
    fp=np.ones(xi.size)
    fp[1:xi.size-1]=(f[J+1]-f[J-1])/(xi[J+1]-xi[J-1])
    fp[0]=(f[0]-fA)/(xi[0]-A)
    fp[-1]=(fB-f[-1])/(B-xi[-1])
    return fp

def cint(xi,f,mu,kap,b):
    return 2*kap*(b-xi)*Fp(f)**2*f**3/(mu+kap*f**4)

def c(f,a,b,mu,kap):     
    return 1/(b-a)*(1+4*b*(np.log(b/a)-1.+a/b) + (2*kap*omga*(b-xi)*Fp(f)**2*f**3/(mu+kap*f**4)).sum(axis=0))
#    return 1+4*b*(np.log(b/a)-1.) + quad(cint,a,b,args=(f,mu,kap,b,))

def G(f,a,b,mu,kap):
    return 1.+c(f,a,b,mu,kap)*(xi-a)-4*a*(xi/a*(np.log(xi/a)-1)+1)

def Ffunc(f,mu,kap):
    return f**3/(mu+kap*f**4)

def iesolve(f,a,b,mu,kap):
    return f-G(f,a,b,mu,kap) + np.array([(1+Qs.ref_nodes[i])*(2*kap*(0.5*(b-a))*Qs.ref_weights*(Qs.ref_nodes[i]-Qs.ref_nodes)*Fp(f)**2*f**3/(mu+kap*f**4)).sum(axis=0) for i in range(xi.size)],float)

def func(r,y):
    f,fp=y
    return np.array([fp,-4./r - 2*Kappa*(f**3*fp**2)/(Mu+Kappa*f**4)])

def bc(fa,fb):
    return np.array([fa[0]-fA,fb[0]-fB])

def sphreplot(a,b):
    ax = plt.figure()
    pot=ax.gca(projection='3d')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    pot.set_xlabel(r'$x$',fontsize=18)
    pot.set_ylabel(r'$y$',fontsize=18)
    pot.set_zlabel(r'$z$',fontsize=18)
    #ax.view_init(azim=330)
    r=np.meshgrid(np.linspace(a,b,100),np.linspace(a,b,100))
    thta=np.linspace(0,2*np.pi,100)
    phi=np.linspace(0,1.5*np.pi,100)
    
    T,P=np.meshgrid(thta,phi)
    X=r*np.sin(T)*np.cos(P)
    Y=r*np.sin(T)*np.sin(P)
    Z=r*np.cos(T)
    col1 = plt.cm.Blues(np.linspace(0,-2,2)) 
    col1 = np.repeat(col1[np.newaxis,:, :], 100, axis=0) 
    return pot.plot_surface(X,Y,Z,rstride=4,cstride=4,color='black',cmap=cm.coolwarm,linewidth=100, antialiased=False, alpha=0.6)

def sol_collocation(n):
    R=np.linspace(A,B,n)
    ya=np.zeros((2,R.size))
    ya[0]=np.linspace(fA,fB,R.size)
    ya[1]=(fB-fA)/R.size*np.ones(R.size)
    colsol = solve_bvp(func,bc,R,ya)
    return colsol

sol=root(iesolve,sol_collocation(200).sol(xi)[0],args=(A,B,Mu,Kappa),method='krylov').x
plt.figure(1)
plt.tick_params(labelsize=12)
plt.plot(xi,sol,label='Nystr\\"{o}m (N=%d)'%(Ns))
plt.xlim(A,B)
plt.ylim(fA,fB)
plt.ylabel(r'$f(R)$',fontsize=12)
plt.xlabel(r'$R$',fontsize=12)
plt.plot(xi,sol_collocation(200).sol(xi)[0],label=r'Collocation ($N_p$ = 100)')
plt.legend(loc=0,fontsize=12)
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.savefig('newtonkrylov.eps')

plt.figure(2)
plt.tick_params(labelsize=12)
plt.plot(xi[:-1],Fp(sol)[:-1],label='Nystr\\"{o}m (N=%d)'%(Ns))
#plt.xlim(A,B)
#plt.ylim(fA,fB)
plt.ylabel(r'$f^{\prime}(R)$',fontsize=12)
plt.xlabel(r'$R$',fontsize=12)
plt.plot(xi,sol_collocation(100).sol(xi)[1],label=r'Collocation ($N_p$ = 100)')
plt.legend(loc=0,fontsize=12)
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

np.savetxt('xi100.dat',xi)
np.savetxt('f100.dat',sol)

#plt.savefig('newtonkrylov1.eps')
