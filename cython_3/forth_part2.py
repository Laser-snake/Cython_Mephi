import numpy as np
from numpy import linalg as LA
import cython
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import random
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})

from numpy import exp, cos, sin, cosh, sinh, sqrt, pi
from matplotlib.animation import FuncAnimation
from numpy import *


class Solver:

  def __init__(self, N, B, J, T, M, var):          #задаем константы
     self.N = N
     self.B = B
     self.J = J
     self.T = T
     self.M = M
     self.var = var

  def massive_s(self):           #задаем массив спинов
   if self.var == 1:                 #одномерный случай
    s = np.random.randint(0, 2, (self.N))
    for i in range(self.N):
     if s[i] == 0:
        s[i] = -1
   if self.var == 2:                 #двумерный случай
    s = np.random.randint(0, 2, (self.N, self.N))
    for i in range(self.N):
     for j in range(self.N):
      if s[i,j] == 0:
         s[i,j] = -1
    #s = np.random.randint(1, 2, self.N)    
   return s

  def summ(self, s):         #суммирование для спинов
   S1 = 0
   S2 = 0
   S3 = 0 

   if self.var == 1:              #одномерный случай
    sum_1 = np.zeros((self.N))
    sum_2 = np.zeros((self.N))
    for i in range(self.N-1):
      sum_2[i] = s[i]*s[i+1]
      S2+= sum_2[i] 
      S1+= s[i] 

   if self.var == 2:                      #двумерный случай
    sum_1 = np.zeros((self.N,self.N))
    sum_2 = np.zeros((self.N,self.N))
    sum_3 = np.zeros((self.N,self.N))
    for i in range(self.N-1):
      for j in range(self.N-1):
        sum_2[i,j] = s[i,j]*s[i+1,j]
        sum_3[i,j] = s[i,j]*s[i,j+1]
        S3+= sum_3[i,j]
        S2+= sum_2[i,j] 
        S1+= s[i,j] 

   return S1, S2, S3

  def energy(self,s):                  #энергия состояния
    if self.var == 1:                    #одномерный случай
      sum1, sum2, sum3 = solve.summ(s)
      Energy = -self.J *sum2 - self.B*sum1
    if self.var == 2:                         #двумерный случай
      sum1, sum2, sum3 = solve.summ(s)
      Energy = -self.J *sum2 - self.J *sum3 - self.B*sum1

    return Energy


  def metropolis(self,s,T):               #алгоритм метрополис
  
   if self.var == 1:                #одномерный случай
   #s = solve.massive_s()   
    j = random.randint(0, self.N)
    #print(j)
    Ek = solve.energy(s)

    s_tr = np.copy(s)  
    s_tr[j] = -s_tr[j]
    Ej = solve.energy(s_tr)
    if Ej <= Ek:
       s_new = np.copy(s_tr)
    else:
      R = np.exp(-(Ej - Ek)/T) 
      #print(R)  
      r = random.random()
      if R < r:
        s_new = np.copy(s)
      else:
        s_new = np.copy(s_tr)
    #print(s_new,'s_new')  
   if self.var == 2:                     #двумерный случай
      j = random.randint(0, self.N)
      i = random.randint(0, self.N)  
      Ek = solve.energy(s)
      s_tr = np.copy(s)
      s_tr[i,j] = -s_tr[i,j]
      Ej = solve.energy(s_tr)
      if Ej <= Ek:
       s_new = np.copy(s_tr)
      else:
       R = np.exp(-(Ej - Ek)/T) 
      #print(R)  
       r = random.random()
       if R < r:
        s_new = np.copy(s)
       else:
        s_new = np.copy(s_tr)
        
   return s_new     


  def plotter_energy(self):       #енергия одномерная
    E = np.zeros(110)
    s = solve.massive_s()
    count = 0
    #print(s)
    for T in range(1,110):
      for i in range(self.M):
        
        count+= solve.energy(s)
        s = solve.metropolis(s, T/10)
      E[T] = count/self.M/100
      count = 0
    t = np.arange(100)

    fig, ax = plt.subplots()
    ax.plot(t/10, E[10:])

    ax.set_xlabel("t")
    ax.set_ylabel("Energy")
    plt.show()
    
  def plotter_energy_2D(self):       #энергия двумерная
   if self.var == 2:       
    E = np.zeros(60)
    s = solve.massive_s()
    count = 0
    #print(s)
    for T in range(1,60):
      for i in range(self.M):
       for j in range(self.M):
        count+= solve.energy(s)
        s = solve.metropolis(s, T/10)
      E[T] = count/self.M/100/self.M/100
      count = 0
    t = np.arange(50)

    fig, ax = plt.subplots()
    ax.plot(t/10, E[10:]*3.2)

    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Energy")
    plt.show()    
    
  def plotter_heatcapacity(self):     
    E = np.zeros(50)
    E_sq = np.zeros(50)
    C = np.zeros(50)
    s = solve.massive_s()
    count = 0
    count2 = 0
    #print(s)
    for T in range(1,50):
      for i in range(self.N):
        
         count+= solve.energy(s)
         count2+= solve.energy(s)*solve.energy(s)
         s = solve.metropolis(s, T/10)
      count = count/self.N   
      count2 = count2/self.N 
      E[T] = count*count
      E_sq[T] = count2
      C[T] = (E_sq[T] - E[T])/T**2
      count = 0
      count2 = 0
      
    T = np.arange(50)
    M = (1/T)**2/(np.cosh(1/T))**2
    fig, ax = plt.subplots()
    ax.plot(T/10, C/2, color='blue')
    ax.plot(T/10, M, color='green', label= r'$C=\frac{(J/kT)^2}{\cosh^2\left(\frac{J}{kT}\right)}$')
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Heat capacity")
    ax.legend()
    plt.show()
    
  def plotter_heatcapacity_2D(self):     
    E = np.zeros(100)
    E_sq = np.zeros(100)
    C = np.zeros(100)
    s = solve.massive_s()
    count = 0
    count2 = 0
    #print(s)
    for T in range(1,100):
      for i in range(self.N):
        
         count+= solve.energy(s)
         count2+= solve.energy(s)*solve.energy(s)
         s = solve.metropolis(s, T/10)
      count = count/self.N   
      count2 = count2/self.N 
      E[T] = count*count
      E_sq[T] = count2
      C[T] = (E_sq[T] - E[T])/T**2
      count = 0
      count2 = 0
    T = np.arange(100)
    M = 2.269185*np.ones(50)
    fig, ax = plt.subplots()
  
    ax.plot(T/10, C, color='blue')
    ax.plot(T/10, M, color='green', label= 'Tc = 2.269185')
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Heat capacity")
    ax.legend()
    plt.show()
   
    
  def plotter_namagn(self):        #намагниченность одномерная
    M = np.zeros(self.M)
    D = np.ones(100)
    s = solve.massive_s()
    count = 0
    
    #print(s)
    for T in range(1,100):  
      s = solve.massive_s()
      for i in range(self.M):
      
       #for j in range(self.N):   
       s = solve.metropolis(s, T/10)
        
       M[i] = abs(sum(s))/self.N
      
      D[T]=sum(M[10*self.N:]) / (self.M - 10*self.N) 
      #print(D[T])
      #D[T]= sum(M) / self.M
      count = 0
    t = np.arange(100)
    
    fig, ax = plt.subplots()
    ax.plot(t/10, D)

    ax.set_xlabel("t")
    ax.set_ylabel("M")
    plt.show()

  def plotter_namagn_2D(self):        #намагниченность двумерная
    M = np.zeros(self.M)
    D = np.ones(100)
    s = solve.massive_s()
    count = 0
    
    #print(s)
    for T in range(1,100):  
      s = solve.massive_s()
      for i in range(self.M):
       s = solve.metropolis(s, T/10)     
       M[i] = abs(sum(sum(s)))/self.N/self.N
      D[T]=sum(M[10*self.N:]) / (self.M - 10*self.N)  
      #print(D[T])
      #D[T]= sum(M) / self.M
      count = 0
    t = np.arange(100)
    
    fig, ax = plt.subplots()
    ax.plot(t/10, D, color ='blue')

    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("M")
    plt.show()


var = 1
potential = 2

N = 20
M = 20
B = 0
J = 1
T = 1
#print(dx,dt)

solve = Solver(N, B, J, T, M, var)
#solve.plotter_energy()
#solve.plotter_namagn()
solve.plotter_heatcapacity()
#solve.plotter_energy_2D()
#solve.plotter_namagn_2D()




