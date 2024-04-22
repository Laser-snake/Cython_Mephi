import numpy as np
cimport cython
cimport numpy as np
ctypedef np.float64_t float64_t
from cython.view cimport array as cvarray
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport pi
#import cmath

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.initializedcheck(False)

#solver.gauss_method(self.u, self.size_system, self.step_system, self.iter)
def gauss_method(np.ndarray[np.float64_t, ndim=2] U,double l, int n_x, int n_t, np.ndarray[np.float64_t, ndim=2] rho):
    cdef double[:, :] matrics = U[:, :]
    cdef double[:, :] matrics_1 = U[:, :]
    cdef int grid_size = n_x
    cdef double size_system = l
    cdef int iterration = n_t
    cdef double [:,:] h = rho[:, :]
    cdef double delta = size_system/grid_size
    cdef double norm = 0.
    cdef double norm_1 = 0.1
    print(delta)
    #cdef int sets = sets

    cdef int i,j,k
    for k in range(0, iterration):
        print((norm - norm_1)/norm_1)
        norm = 0.
        norm_1 = 0.1
        for i in range(1,grid_size-1):
            for j in range(1,grid_size-1):
                matrics_1[i,j] = 0.25*(matrics[i-1,j] + matrics[i,j-1] + matrics[i+1,j] + matrics[i,j+1]) + pi * h[i,j] * delta ** 2

        for i in range(1,grid_size-1):
            for j in range(1,grid_size-1):
                norm += np.abs(matrics_1[i,j] - matrics[i,j]) ** 2
            for j in range(1,grid_size-1):
                matrics[i ,j] = matrics_1[i,j]
        print(norm)

    return np.asarray(matrics)



def gauss_zedel_method(np.ndarray[np.float64_t, ndim=2] U,double l, int n_x, int n_t, np.ndarray[np.float64_t, ndim=2] rho):
    cdef double[:, :] matrics = U[:, :]
    cdef double[:, :] matrics_1 = U[:, :]
    cdef int grid_size = n_x
    cdef double size_system = l
    cdef int iterration = n_t
    cdef double [:,:] h = rho[:, :]
    cdef double delta = size_system/grid_size
    cdef double norm = 0.
    cdef double norm_1 = 0.1
    print(delta)
    #cdef int sets = sets

    cdef int i,j,k
    for k in range(0, iterration):
        print((norm - norm_1)/norm_1)
        norm = 0.
        norm_1 = 0.1
        for i in range(1,grid_size-1):
            norm = norm + np.abs(matrics[i,i])
            for j in range(1,grid_size-1):
                matrics[i,j] = 0.25*(matrics[i-1,j] + matrics[i,j-1] + matrics[i+1,j] + matrics[i,j+1]) + pi * h[i,j] * delta ** 2
        for i in range(1,grid_size-1):
            norm_1 = norm_1 + np.abs(matrics[i,i])

    return np.asarray(matrics)


def gauss_zedel_relax_method(np.ndarray[np.float64_t, ndim=2] U,double l, int n_x, int n_t, np.ndarray[np.float64_t, ndim=2] rho, double omega_0):
    cdef double[:, :] matrics = U[:, :]
    cdef double[:, :] matrics_1 = U[:, :]
    cdef int grid_size = n_x
    cdef double size_system = l
    cdef int iterration = n_t
    cdef double [:,:] h = rho[:, :]
    cdef double delta = size_system/grid_size
    cdef double norm = 0.
    cdef double norm_1 = 0.1
    cdef double delta_matrics = 0.1
    cdef double omega = omega_0
    print(delta)
    #cdef int sets = sets

    cdef int i,j,k
    for k in range(0, iterration):
        print((norm - norm_1)/norm_1)
        norm = 0.
        norm_1 = 0.1
        for i in range(1,grid_size-1):
            norm = norm + np.abs(matrics[i,i])
            for j in range(1,grid_size-1):
                delta_matrics = 0.25*(matrics[i-1,j] + matrics[i,j-1] + matrics[i+1,j] + matrics[i,j+1]) + pi * h[i,j] * delta ** 2
                delta_matrics = delta_matrics - matrics[i, j]
                matrics[i, j] =  matrics[i, j] + omega * delta_matrics
        for i in range(1,grid_size-1):
            norm_1 = norm_1 + np.abs(matrics[i,i])

    return np.asarray(matrics)
