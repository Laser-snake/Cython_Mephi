import numpy as np
import cython
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import random
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
import file






class Phys:
    K  = 1.0
    C = 1.
    density = 1.
    a = (K)/(C* density)


class SolverHeatCondactivityEquation(Phys):
    def __init__(self, length_system = 1, final_time = 1., steps = 100., n_x = 100, n_t=100, var = 1, method = 'FTCS'):
        self.length_system = length_system
        self.final_time = final_time
        self.n_x = n_x
        self.n_t = n_t
        self.dx = length_system/n_x
        self.dt = final_time/n_t
        self.steps = steps
        self.u = None
        alfa = self.a**2 * (self.dt)/(self.dx**2)
        if var == 1:
            if alfa <= 1/2:
                print(f'alphe:{alfa}')
                u = self.condition_1()
                self.plotter(u)
                u = self.solver(u, alfa)
                self.plotter(u)

            else:
                print(f"Схема не будет сходится измените шаг сетки:{alfa}")
        elif var == 2:
            if alfa <= 1/2:
                print(f'alphe:{alfa}')
                u = self.condition_2()
                self.plotter(u)
                u = self.solver(u, alfa)
                self.plotter(u)
            else:
                print(f"Схема не будет сходится измените шаг сетки:{alfa}")
        elif var == 3:
            u = self.condition_1()
            self.plotter(u)
            u = self.solver(u, alfa, method)
            self.plotter(u)
        elif var == 4:
            u = self.condition_2()
            self.plotter(u)
            u = self.solver(u, alfa, method)
            self.plotter(u)
        elif var == 5:
            u = self.condition_3()
            self.plotter(u)
            u = self.solver(u, alfa, method)
            self.plotter(u)



    def condition_1(self):
        u = 50*np.zeros((self.n_x, self.n_t))
        u[:, 0] = 100
        return u

    def condition_2(self):
        u = np.zeros((self.n_x, self.n_t))
        x = np.linspace(0, self.length_system, self.n_x)
        u[:, 0] = np.sin(np.pi * x/ self.length_system)
        return u

    def condition_3(self):
        u = np.zeros((self.n_x, self.n_t))
        x = np.linspace(0, self.length_system, self.n_x)
        u[0:int(self.n_x/2), 0] = 50.
        u[int(self.n_x/2):self.n_x, 0] = 100.
        return u

    def stepper(self,u ,alfa ,method = 'FTCS'):
        if method == 'FTCS':
            return file.FTCS(u, self.n_x, self.n_t, alfa)
        if method == 'CrankNicholson':
            print('cr')
            k1 = np.zeros((self.n_x, self.n_x))
            k2 = np.zeros((self.n_x, self.n_x))
            return file.CrankNicholson(u, self.n_x, self.n_t, alfa, k1, k2)
        if method == 'FTCS_change':
            uc = 50.
            h = 0.1
            return file.FTCS_change(u, self.n_x, self.n_t, alfa, uc, h, self.dt)

    def plotter(self, u):
        ax = plt.figure().add_subplot(projection='3d')
        x = np.linspace(0, self.length_system, self.n_x)
        t = np.linspace(0, self.final_time, self.n_t)
        T, X = np.meshgrid(t, x)
        print(u.shape)
        U = u
        ax.plot_surface(X, T, U, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,alpha=0.3)

        #ax.contour(X, T, U, zdir='z', cmap='coolwarm')
        #ax.contour(X, T, U, zdir='x', cmap='coolwarm')
        #ax.contour(X, T, U, zdir='y', cmap='coolwarm')

        ax.set(xlabel='X', ylabel='t', zlabel='Potential u')

        plt.show()


    def solver(self, u ,alfa ,method = 'FTCS'):
        u = self.stepper(u ,alfa , method)
        return u


#__init__(self, length_system = 1, final_time = 1., steps = 100., n_x = 100, n_t=100, var = 1, method = 'FTCS')

n_x = 100
steps = 10000
n_t = 5000
L = 100
t = 100
var = 3
a = SolverHeatCondactivityEquation(L, t, steps, n_x, n_t, var, 'FTCS_change')
