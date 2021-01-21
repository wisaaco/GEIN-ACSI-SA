#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:37:14 2020

@author: isaaclera
"""
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# =============================================================================
# ""REGRESION LINEAL"""
#               f(x) = a  + b  x
# x - representa el eje del tiempo, 
# y - presenta el valor de interes (número de usuarios, clientes, peticiones,...)
# a - es el punto de corte con algun eje x o y (INTERCEPT)
# b - es la pendiente de la recta (SLOPE)
# =============================================================================
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


x = np.random.randint(low=3,high=100, size=40)
print("X: %s"%x)

y = np.random.randint(low=3,high=100, size=40)
print("Y: %s"%y)


slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("slope: %f    intercept: %f" % (slope, intercept))


print("R-squared: %f" % r_value**2)


plt.plot(x, y, 'o', label='original data')
plt.plot(x, intercept + slope*x, 'r', label='fitted line')
plt.legend()
plt.show()

# =============================================================================
# Sabiendo A y B podemos aplicar la función a cualquier otro valor temporal de x
#               f(120) = a+b*120 
#               f(300) = 
#               f(239093029) = 
#
# A mayor distanciamiento, el error puede ser mayor!
# =============================================================================


# =============================================================================
#  MEDIA MOVILES
# Las medias moviles permiten calcular un punto 'medio' en función de un registro histógico
# Por ejemplo, con una ventana de 3 registros, en el instante siguiente podemos usar la media 
# de esos 3 para calcular el siguiente.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
 
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 
y = [3,5,1,8,2,1,6,1,9,2]
x = range(len(y))

plt.plot(x,y)
plt.show()

#1 Caso con pocas muestras
windowsize = 3
yMA = movingaverage(y,windowsize)
print(yMA)

plt.plot(x,y)
plt.plot(range(windowsize,len(yMA)+windowsize),yMA)
plt.show()


#2 Caso con mayor número de registros
y = np.random.randint(low=3,high=100, size=100)
x = range(len(y))

yMA = movingaverage(y,10)
 
plt.plot(x,y)
plt.plot(range(windowsize,len(yMA)+windowsize),yMA)
plt.show()
