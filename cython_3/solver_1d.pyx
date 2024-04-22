import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange

from cython.view cimport array as cvarray
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport pi
from libc.stdlib cimport rand, RAND_MAX
import cmath

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double energy(double[:] alpha, int size_chain, double j, double mu, double b_field) nogil:
    cdef double [:] matrics = alpha[:]
    cdef double j_chain = j
    cdef double mu_chain = mu
    cdef double b_chain = b_field
    cdef int n = size_chain
    cdef int i
    cdef double e = 0
    for i in range(n-1):
        e = -j_chain * matrics[i] * matrics[i+1] + e
    for i in range(n):
        e = -b_chain*mu_chain * matrics[i] + e
    return e

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:,:] metropolis(double[:,:] alpha, int size_chain, int iteration, double temp,double J, double mu, double B) nogil:
    cdef double [:,:] matrics = alpha[:,:]
    cdef double J_chain = J
    cdef double mu_chain = mu
    cdef double B_chain = B
    cdef int n = size_chain
    cdef int iterat = iteration
    cdef double T = temp
    cdef int i,j,k,m,v
    cdef double e_old, e_new,r,R, y,e,e_1
    for j in range(1,iterat):
        m = rand() % (n - 0)
        e_old = energy(matrics[:,j-1], n, J_chain, mu_chain, B_chain)
        for i in range(n):
            y = matrics[i,j-1]
            matrics[i,j] = y
        y = -matrics[m,j-1]
        matrics[m,j] = y
        e_new = energy(matrics[:,j], n, J_chain, mu_chain, B_chain)
        if e_new > e_old:
             r = rand()/float(RAND_MAX)
             R = exp(-(e_new -e_old)/T)
             if R < r:
                y = -matrics[m,j]
                matrics[m,j] = y
    return matrics


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def metropolis_start(np.ndarray[dtype=double, ndim=3] alpha, np.ndarray[dtype=double, ndim=2] betta, np.ndarray[dtype=double, ndim=1] gamma,  int size_chain, int iteration, double temp,double temp_end, int size_temprature,double J, double mu, double B):
    cdef double [:,:,:] matrics = alpha[:,:,:]
    cdef double [:,:,:] matrics_end = alpha[:,:,:]
    cdef double [:,:] matrix_e = betta[:,:]
    cdef double [:] matrix_t = gamma[:]
    cdef double J_chain = J
    cdef double mu_chain = mu
    cdef double B_chain = B
    cdef int n = size_chain
    cdef int iterat = iteration
    cdef double T = temp
    cdef double T_end = temp_end
    cdef int n_t = size_temprature
    cdef double dT = (temp_end - temp)/n_t
    cdef int i
    with nogil:
        for i in prange(n_t):
            matrics_end[:,:,i] = metropolis(matrics_end[:,:,i], n, iterat, matrix_t[i],J_chain, mu_chain, B_chain)
    return np.asarray(matrics_end), np.asarray(matrix_e)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_mag(np.ndarray[dtype=double, ndim=3] alpha, np.ndarray[dtype=double, ndim=1] betta, int size_temprature, int iter, int size_chain):
    cdef double [:,:,:] matrics = alpha[:,:,:]
    cdef double [:] matrix_av = betta[:]
    cdef double mag = 0.
    cdef double mag_av = 0.
    cdef int i,j,k
    for j in range(size_temprature):
        mag_av = 0.
        for i in range(size_chain * 10):
            mag = 0.
            for k in range(size_chain):
                mag += matrics[k,iter-size_chain * 10 + i -1,j]
            mag = mag/size_chain
            mag_av += mag
        mag_av = mag_av/(size_chain * 10)
        matrix_av[j] = mag_av
    return np.asarray(matrix_av)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_e(np.ndarray[dtype=double, ndim=3] alpha, np.ndarray[dtype=double, ndim=1] betta, int size_temprature, int iter, int size_chain, double J, double mu, double B):
    cdef double [:,:,:] matrics = alpha[:,:,:]
    cdef double [:] matrix_av = betta[:]
    cdef double mag = 0.
    cdef double mag_av = 0.
    cdef int i,j,k
    for j in range(size_temprature):
        mag_av = 0.
        for i in range(size_chain * 10):
            #iter - size_chain * 10 + i-1
            mag = energy(matrics[:,iter - size_chain * 10 + i-1, j], size_chain, J, mu, B)
            mag_av += mag
        mag_av = mag_av/(size_chain * 10)
        matrix_av[j] = mag_av
    return np.asarray(matrix_av)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_cv(np.ndarray[dtype=double, ndim=3] alpha, np.ndarray[dtype=double, ndim=1] betta, np.ndarray[dtype=double, ndim=1] gamma, int size_temprature, int iter, int size_chain, double J, double mu, double B):
    cdef double [:,:,:] matrics = alpha[:,:,:]
    cdef double [:] matrix_av = betta[:]
    cdef double mag = 0.
    cdef double mag_av = 0.
    cdef double mag_2 = 0.
    cdef double mag_av_2 = 0.
    cdef int i,j,k
    for j in range(size_temprature):
        mag_av = 0.
        mag_av_2 = 0.
        for i in range(size_chain * 10):
            #iter - size_chain * 10 + i-1
            mag = energy(matrics[:,iter - size_chain * 10 + i-1, j], size_chain, J, mu, B)
            mag_av += mag
            mag_av_2 += mag**2
        mag_av = mag_av/(size_chain * 10)
        mag_av_2 = mag_av_2/(size_chain * 10)
        matrix_av[j] = (mag_av_2 - mag_av**2)/gamma[j] ** 2
    return np.asarray(matrix_av)
