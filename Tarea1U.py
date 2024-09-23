import math
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sympy import symbols
import matplotlib.pyplot as plt

#--------------TABLA DE VALORES BI, LAMBDA1 y A1 PARA UNA ESFERA-------------
tabla=np.array([
            [0.01, 0.1730, 1.0030],
            [0.02, 0.2445, 1.0060],
            [0.04, 0.3450, 1.0120],
            [0.06, 0.4217, 1.0179],
            [0.08, 0.4860, 1.0239],
            [0.1, 0.5423, 1.0298],
            [0.2, 0.7593, 1.0592],
            [0.3, 0.9208, 1.0880],
            [0.4, 1.0528, 1.1164],
            [0.5, 1.1656, 1.1441],
            [0.6, 1.2644, 1.1713],
            [0.7, 1.3525, 1.1978],
            [0.8, 1.4320, 1.2236],
            [0.9, 1.5044, 1.2488],
            [1.0, 1.5708, 1.2732],
            [2.0, 2.0288, 1.4793],
            [3.0, 2.2889, 1.6227],
            [4.0, 2.4556, 1.7202],
            [5.0, 2.5704, 1.7870],
            [6.0, 2.6537, 1.8338],
            [7.0, 2.7165, 1.8673],
            [8.0, 2.7654, 1.8920],
            [9.0, 2.8044, 1.9106],
            [10.0, 2.8363, 1.9249],
            [20.0, 2.9857, 1.9781],
            [30.0, 3.0372, 1.9898],
            [40.0, 3.0632, 1.9942],
            [50.0, 3.0788, 1.9962],
            [100.0, 3.1102, 1.9990]
            ])
Bi=tabla[:,0]
lambda1=tabla[:,1]
A1=tabla[:,2]
#-----------------------DATOS CONOCIDOS A INGRESAR---------------------------------
per=float(0.21)
vol=float(90)
masa=float(134)
x0=float(1.0)
Ti=float(-3.9)
Tamb=float(19.1)
Cp=float(3620)
#----------------------TEMPERATURAS y TIEMPO A INGRESAR-------------------------
T0=[-3.9,-3.8,-3.8,-3.5,-3.1,-3.1,-2.8,-2.8,-2.6,-2.4,-2.2,-1.9,-1.5]
Tr=[5.1,5.2,5.4,5.8,6.2,6.4,6.9,7.3,8.5,9.2,9.7,9.9,10.5]
tiempo=[1,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600]
num_temp=len(T0)
#------------------------CREACIÓN DE LAS LISTAS VACIAS--------------------------
lambdas=[]
Bis=[]
A1s=[]
taus=[]
alphas=[]
ks=[]
hs=[]
#----------------------------SIMBOLOS--------------------------------
a='\u03B1'
L='\u03BB'
T='\u03C4'
# -------------------------METODO DE NEWTON RAPHSON--------------------------
def f(x, cte):
    return math.sin(x)/x - cte
def df(x):
    return(x*math.cos(x)-math.sin(x))/(x**2)
def newton_raphson(cte, x0, tol=1e-8, max_iter=100):
    iteracion=0
    while iteracion < max_iter:
        fx=f(x0,cte)
        if abs(fx) < tol:
            return x0
        x1 = x0 -fx / df(x0)
        x0=x1
        iteracion += 1
    print("El método no converge despues de", max_iter, "iteraciones.")
    return None
#------------HALLAR EL RADIO DE LA PERA MEDIANTE EL PERÍMETRO---------------
def radPer(p):
    r=p/(2*math.pi)
    return r
ro=radPer(per)
#-----------------------BUCLE PARA BI, LAMBDA y A1-------------------------
i=0
while i < num_temp:
    #----------------------HALLAR LAMBDAS-------------------------------
    theta=(Tr[i]-Tamb)/(T0[i]-Tamb)
    lamb=newton_raphson(theta,x0)
    lambdas.append(round(lamb,5))
    #-------------------HALLAR NÚMEROS BI-------------------------------
    f_interBi=interp1d(lambda1, Bi, kind='linear', fill_value="extrapolate")
    Binterp=f_interBi(lambdas[i]).item()
    Bis.append(round(Binterp,4))
    #-----------------------HALLAR A1's----------------------------------
    f_interA1=interp1d(lambda1, A1, kind='linear', fill_value="extrapolate")
    A1nterp=f_interA1(lambdas[i]).item()
    A1s.append(round(A1nterp,5))
    i += 1
#-----------------BUCLE PARA HALLAR TAU, APLHA, K y H---------------------
j=0
p=(0.001*masa)/(0.000001*vol)
while j < num_temp:
    #-------------------------HALLAR TAU--------------------------------
    tau=(math.log(((T0[j]-Tamb)/(Ti-Tamb))/A1s[j]))/(-(lambdas[j])**2)
    taus.append(round(tau,5))
    #------------------------HALLAR ALPHA--------------------------------
    alpha=(tau*(ro**2))/tiempo[j]
    alphas.append(round((alpha),9))
    #--------------------------HALLAR K---------------------------------
    k=p*Cp*alpha
    ks.append(round(float(k),5))
    #--------------------------HALLAR H--------------------------------
    h=(ks[j]*Bi[j])/ro
    hs.append(round(float(h),5))
    j+=1
#-----------------BUCLE PARA CALCULAR LOS VALORES PROMEDIOS------------
m=1
sumH=0
sumK=0
sumAlpha=0
while m < num_temp:
    sumH=(sumH+hs[m])
    sumK=(sumK+ks[m])
    sumAlpha=(sumAlpha+alphas[m])
    m += 1
#--------------------------CREAR GRÁFICOS-------------------------
plt.figure(1)
plt.plot(tiempo, T0, marker='o')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura en el centro [°C]')
plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

plt.figure(2)
plt.plot(tiempo,Tr, marker='o')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura en la superficie [°C]')
plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

plt.figure(3)
plt.plot(tiempo, ks, marker='o')
plt.xlabel('Tiempo [s]')
plt.ylabel('Conductividad térmica k [W/m°C]')
plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

plt.figure(4)
plt.plot(tiempo,alphas, marker='o')
plt.xlabel('Tiempo [s]')
plt.ylabel("Difusividad térmica [m2/s]")
plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

plt.figure(5)
plt.plot(tiempo,hs, marker='o')
plt.xlabel('Tiempo [s]')
plt.ylabel("Coeficiente de convección [W/m2°C]")
plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

plt.figure(6)
plt.plot(Tr,ks, marker='o')
plt.xlabel('Temperatura en la superficie [°C]')
plt.ylabel("Conductividad térmica k [W/m°C]")
plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)
#----------------------IMPRIMIR RESULTADOS-----------------------------
print("Valores de",L,":",lambdas)
print("Valores de Bi : ",Bis)
print("Valores de A1 : ",A1s)
print("Valores de",T,":",taus)
print("Valores de",a,":",alphas,"\n",a,"prom=",sumAlpha/12)
print("Valores de h : ",hs,"\n h prom=",sumH/12)
print("Valores de k : ",ks,"\n k prom=",sumK/12)
print("NUEVO MENSAJE")
plt.show()