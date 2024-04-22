import numpy as np
cimport cython
cimport numpy as np

from cython.view cimport array as cvarray
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport pi
from libc.math cimport random
import cmath


@cython.boundscheck(False)
@cython.wraparound(True)
@cython.initializedcheck(False)
#@cython.cdivision(True)


def solver(np.ndarray[dtype=complex, ndim=2] U, int n, int sets, complex alfa,np.ndarray[dtype=complex, ndim=2] k1,np.ndarray[dtype=complex, ndim=2] k2, np.ndarray[dtype=complex, ndim=1] pote):
   cdef complex[:, :] matrics = U[:, :]
   cdef complex[:,:] k_1 = k1[:, :]
   cdef complex[:, :] k_2 = k2[:, :]
   cdef complex [:]v = pote[:]
   cdef int n_x = n
   cdef int n_t = sets
   cdef complex alfa_0 = alfa
   cdef int i,n_iter
   cdef complex A = -1, C = -1
   cdef complex B = 2/alfa_0+2
   #print(1)
   for n_iter in range (0,n_t-1):
      k_1[0, 0] = -C/(B-1/alfa_0 *pote[1])
      k_2[0, 0] = (matrics[0,n_iter+1] + matrics[0,n_iter] + (2/alfa_0-2 + 1/alfa_0 *pote[1])*matrics[1,n_iter] + matrics[2,n_iter])/(B- 1/alfa_0 *pote[1])
      for i in range(n_x-2):
         k_1[i+1, i+1] = -C/(A*k_1[i, i] +B - 1/alfa_0 *pote[i])
         k_2[i+1, i+1] = (matrics[i-1,n_iter] + (2/alfa_0-2 + 1/alfa_0 *pote[i])*matrics[i,n_iter] + matrics[i+1,n_iter] - A * k_2[i, i])/(A * k_1[i, i] +B- 1/alfa_0 *pote[i])
      k_2[n_x-2, n_x-2] = (matrics[n_x-3,n_iter] + (2/alfa_0-2 +1/alfa_0 *pote[n_x-2])*matrics[n_x-2,n_iter] + matrics[n_x-1,n_iter] + matrics[n_x-1,n_iter+1] - A * k_2[n_x-2, n_x-2])/(A * k_1[n_x-2, n_x-2] +B-1/alfa_0 *pote[n_x-2])
      matrics[n_x-2,n_iter+1] = k_2[n_x-2, n_x-2]
      for i in range(n_x-2,0,-1):
         matrics[i,n_iter+1] = k_1[i+1, i + 1]*matrics[i+1,n_iter+1] + k_2[i+1, i+1]
   k_1[:] = 0
   k_2[:] = 0
   return np.asarray(matrics)
