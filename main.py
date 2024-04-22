import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cython
from scipy.special import erf
import scipy
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
import solver


class SolverPoisson():
    def __init__(self, size_system, step_system, boundary_conduction, charge, iteration=1000, method = 'gauss'):
        self.size_system = size_system
        self.charge = 1.0
        self.step_system = step_system
        self.u = boundary_conduction
        self.charge = charge
        self.iter = iteration
        self.u_end = self.u
        self.plotter()
        self.u_end = self.solver_cython(method)
        self.plotter()
        self.field()

    def solver_cython(self, method = 'gauss', omega =1.2):
        if method == 'gauss':
            return solver.gauss_method(self.u, self.size_system, self.step_system, self.iter, self.charge)
        elif method == 'gauss_zedel':
            return solver.gauss_zedel_method(self.u, self.size_system, self.step_system, self.iter, self.charge)
        elif method == 'gauss_zedel_relax':
            return solver.gauss_zedel_relax_method(self.u, self.size_system, self.step_system, self.iter, self.charge, omega)
        else:
            print('non method')

    def plotter(self):
        ax = plt.figure().add_subplot(projection='3d')
        x = np.linspace(0, self.size_system, self.step_system)
        y = np.linspace(0, self.size_system, self.step_system)
        Y, X = np.meshgrid(y, x)
        print(u.shape)
        U = self.u_end
        ax.plot_surface(X, Y, U, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,alpha=0.3)

        #ax.contour(X, T, U, zdir='z', cmap='coolwarm')
        #ax.contour(X, T, U, zdir='x', cmap='coolwarm')
        #ax.contour(X, T, U, zdir='y', cmap='coolwarm')

        ax.set(xlabel='X', ylabel='Y', zlabel='Potential u')

        plt.show()

    def field(self):
        x = np.linspace(0, self.size_system, self.step_system)
        y = np.linspace(0, self.size_system, self.step_system)
        gradient_x, gradient_y = np.gradient(-self.u_end, x[1]-x[0], x[1]-x[0])
        x = np.linspace(0, self.size_system, self.step_system)
        y = np.linspace(0, self.size_system, self.step_system)
        Y, X = np.meshgrid(y[::5], x[::5])
        print(u.shape)
        gradient_x, gradient_y = np.gradient(-self.u_end, x[1]-x[0], x[1]-x[0])
        gradient_x = gradient_x[::5, ::5]
        gradient_y = gradient_y[::5, ::5]
        print(gradient_x.shape)
        plt.quiver(X, Y, gradient_x, gradient_y, linewidth=6)
        plt.show()



charge = np.zeros((100, 100))
charge[10:90,10:11] = 100.
charge[10:90,90:91] = -100.
u = np.zeros((100, 100))

#__init__(self, size_system, step_system, boundary_conduction, charge, iteration=1000, method = 'gauss'):
a = SolverPoisson(10, 100, u, charge, 300, method = 'gauss_zedel')
