import numpy as np
cimport cython
cimport numpy as np

from cython.view cimport array as cvarray
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport pi
import cmath


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)


def FTCS(np.ndarray[np.float64_t, ndim=2] u, int n, int steps, double alfa_0):
   cdef double[:, :] matrics = u[:, :]
   cdef int n_x = n
   cdef int n_t = steps
   cdef double alfa = alfa_0

   cdef int i,j
   for j in range(0,n_t-1):
    for i in range(1,n_x-1):
        matrics[i,j+1] = matrics[i,j] + alfa*(matrics[i-1,j] - 2*matrics[i,j] + matrics[i+1,j])

   #for i in range(1,n_x-1):
     #for j in range(1,n_t-1):
     # matrics[i,j] = matrics[i,j] + alfa*(matrics[i,j-1] - 2*matrics[i,j] + matrics[i,j+1])

   return np.asarray(matrics)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)

def crank_nicholson(np.ndarray[np.float64_t, ndim=2] U, int n, int sets, double alfa,np.ndarray[np.float64_t, ndim=2] k1,np.ndarray[np.float64_t, ndim=2] k2):
   cdef double[:, :] matrics = U[:, :]
   cdef double[:,:] k_1 = k1[:, :]
   cdef double[:, :] k_2 = k2[:, :]
   cdef int n_x = n
   cdef int n_t = sets
   cdef double alfa_0 = alfa
   cdef int i,n_iter
   cdef double A = -1, C = -1
   cdef double B = 2/alfa_0+2


   for n_iter in range (0,n_t-1):
      k_1[0, 0] = -C/(B)
      k_2[0, 0] = (matrics[0,n_iter+1] + matrics[0,n_iter] + (2/alfa_0-2)*matrics[1,n_iter] + matrics[2,n_iter])/( B)
      for i in range(n_x-2):
         k_1[i+1, i+1] = -C/(A*k_1[i, i] +B)
         k_2[i+1, i+1] = (matrics[i-1,n_iter] + (2/alfa_0-2)*matrics[i,n_iter] + matrics[i+1,n_iter] - A * k_2[i, i])/(A * k_1[i, i] +B)
      k_2[n_x-2, n_x-2] = (matrics[n_x-3,n_iter] + (2/alfa_0-2)*matrics[n_x-2,n_iter] + matrics[n_x-1,n_iter] + matrics[n_x-1,n_iter+1] - A * k_2[n_x-2, n_x-2])/(A * k_1[n_x-2, n_x-2] +B)
      matrics[n_x-2,n_iter+1] = k_2[n_x-2, n_x-2]
      for i in range(n_x-2,0,-1):
         matrics[i,n_iter+1] = k_1[i+1, i + 1]*matrics[i+1,n_iter+1] + k_2[i+1, i+1]
   return np.asarray(matrics)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)


def FTCS_change(np.ndarray[np.float64_t, ndim=2] u, int n, int steps, double alfa_0, double uc, double h, double dt):
   cdef double[:, :] matrics = u[:, :]
   cdef int n_x = n
   cdef int n_t = steps
   cdef double alfa = alfa_0
   cdef double u_c = uc
   cdef double h_c = h
   cdef double dt_c = dt

   cdef int i,j
   for j in range(0,n_t-1):
    for i in range(1,n_x-1):
        matrics[i,j+1] = matrics[i,j] + alfa*(matrics[i-1,j] - 2*matrics[i,j] + matrics[i+1,j]) - (matrics[i,j] - u_c) * h_c * dt_c
   return np.asarray(matrics)
