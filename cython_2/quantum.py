import numpy as np
import matplotlib.pyplot as plt
import cython
from scipy.special import erf
import scipy
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
import crank_nicolson

class Phys:
    # класс физических констант
    hbar = 1#0.6582119                   # ps* meV
    me = 1                       # Electron mass in ps^2 meV / micron^2
    m_LP = 1#9.2*10**(-5)*me#9.2*10*(-5)*me#7.2*10**(-5)*me       # Polariton mass
    alpha_1 = 2*10**(-3)#5*10**(-4)#10*10**(-4)                # meV micron^2 (this is actually hbar*u)
    alpha_2 = 0#-.1 * alpha_1         # meV micron^2 (this is actually hbar*u)


class ParamsSystem(Phys):
    def __init__(self, time_steps, space_steps, light_system=0.0, final_time=0.0):
        self.time_steps = time_steps
        self.final_time = final_time
        self.space_steps = space_steps
        self.light_system = light_system
        self.psi_0 = np.zeros(space_steps)
        self.V_pote = np.zeros(space_steps)
        self.grid_k = (2 * np.pi)/light_system * np.fft.fftfreq(space_steps, 1./space_steps)
        self.grid_x = np.linspace(-light_system/2.0, light_system/2.0,
                                  space_steps)
        self.dt = final_time/time_steps
        self.dx = self.grid_x[1] - self.grid_x[0]
        self.dk = self.grid_k[1] - self.grid_k[0]
        self.alpha = (-self.hbar ** 2 * self.dt)/(2 * self.m_LP * 1j * self.hbar * self.dx ** 2 )
        print(f'alpha:{self.alpha}')


    def potention(self, amplitude):
        self.V_pote = amplitude * (np.sign(self.grid_x)+1)/2.0
        self.amp = amplitude
        # np.where(self.grid_x > 0, 1, self.V_pote)

    def potention_well(self, amplitude, a):
        self.V_pote = amplitude * ((erf(self.grid_x-a)+1)/2.0 + (erf(-self.grid_x)+1)/2.0)
        self.amp = amplitude
        # np.where(self.grid_x > 0, 1, self.V_pote)

    def quadratic_potential(self, omega):
        self.V_pote = ((self.m_LP * omega ** 2)/2) * (self.grid_x ** 2)
        self.omega = omega

    def inition_state_psi_gauss(self, d, p0, a):
        self.psi_0 = 1/((np.pi * d **2)**(1./4.)) * np.exp(1j*p0 * self.grid_x / self.hbar) * np.exp(-((self.grid_x-a)**2)/(2*d**2))
        self.param = np.array([self.time_steps, self.final_time, self.space_steps, self.light_system, self.hbar, self.m_LP, d, p0, a])
        np.savetxt(f'parameters_well.csv',self.param, delimiter=',')

    def inition_state_psi_well(self, A, a):
        # A = np.sqrt((5*a)/16)
        A = 1/np.sqrt((5*a)/16)
        self.psi_0 = A * np.sin((np.pi * self.grid_x)/a)**3  * (1-((np.sign(self.grid_x-a)+1)/2.0 + (np.sign(-self.grid_x)+1)/2.0))
        self.param = np.array([self.time_steps, self.final_time, self.space_steps, self.light_system, self.hbar, self.m_LP, A, a])
        np.savetxt(f'parameters_well{int(a)}.csv',self.param, delimiter=',')

    def inition_state_psi_quadratic_potential(self, omega, A, B):
        # A **2 + B ** 2 = 1
        # галицкий ||.2 см
        self.psi_0 = A * ((self.m_LP * omega)/(np.pi * self.hbar)) ** (0.25) * np.exp(-((self.m_LP * omega * self.grid_x ** 2)/(2 * self.hbar))) + B * (1/np.sqrt(2)) *((self.m_LP * omega)/(np.pi * self.hbar)) ** (0.25) * np.exp(-((self.m_LP * omega * self.grid_x ** 2)/(2 * self.hbar))) * np.sqrt((self.m_LP * omega)/(self.hbar)) * 2 * self.grid_x
        self.param = np.array([self.time_steps, self.final_time, self.space_steps, self.light_system, self.hbar, self.m_LP, omega, A, B])
        np.savetxt(f'parameters_quadratic_potential{int(omega)}.csv', self.param, delimiter=',')

    def double_well(self, a=15, d=10,amp=50., delta=0):
        self.V_pote = delta+amp * ((np.sign(self.grid_x+d/2)+1)/2.0 - (np.sign(self.grid_x-d/2)+1)/2.0 +(np.sign(-self.grid_x-d/2-a)+1)/2.0 + (np.sign(self.grid_x-d/2-a)+1)/2.0)








def norm_psi(psi, x):
    return np.sqrt(scipy.integrate.simps(abs(psi) ** 2, x))


class WaveFunctionEvolution(Phys):
    def __init__(self, param_class):
        self.time_steps = param_class.time_steps
        self.final_time = param_class.final_time
        self.psi_x_t = np.zeros((param_class.space_steps, self.time_steps), dtype=complex) + 0.000000000001
        self.psi_x_t[:, 0] = param_class.psi_0
        #solver(np.ndarray[dtype=complex, ndim=2] U, int n, int sets, complex alfa,np.ndarray[dtype=complex, ndim=2] k1,np.ndarray[dtype=complex, ndim=2] k2):
        k1 = np.zeros((param_class.space_steps, param_class.space_steps), dtype=complex)
        k2 = np.zeros((param_class.space_steps, param_class.space_steps), dtype=complex)
        pote = param_class.V_pote * (param_class.dt)/( 1j * self.hbar)
        self.psi_x_t_res = crank_nicolson.solver(self.psi_x_t, int(param_class.space_steps), int(self.time_steps), param_class.alpha, k1, k2,pote)
        np.savetxt(f'psi_x_t.csv',self.psi_x_t_res, delimiter=',')



class SimulatedSystems():
    def free_movement(self):
        # свободный случай
        # __init__(self, time_steps, space_steps, light_system=0.0, final_time=0.0):
        a = ParamsSystem(3000, 1000, 140, 40)
        a.potention(0.)
        # def inition_state_psi_gauss(self, d, p0, a):
        a.inition_state_psi_gauss(6, 1.58, -7)
        #plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, np.abs(a.psi_0)**2)
        plt.show()
        b = WaveFunctionEvolution(a)
        b.solver(a)
        plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, np.abs(b.psi_x_t[3000-1])**2)
        plt.show()

    def potential_step(self):
        # ступенька
        # __init__(self, time_steps, space_steps, light_system=0.0, final_time=0.0):
        a = ParamsSystem(2500, 1000, 240, 30)
        a.potention(3)
        # def inition_state_psi_gauss(self, d, p0, a):
        a.inition_state_psi_gauss(14, 4.2, -38)
        #plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, np.abs(a.psi_0)**2)
        plt.show()
        b = WaveFunctionEvolution(a)
        b.solver(a)
        plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, np.abs(b.psi_x_t[1500-1])**2)
        plt.show()

    def potential_endless_pit(self):
        # яма
        # __init__(self, time_steps, space_steps, light_system=0.0, final_time=0.0):
        a = ParamsSystem(7500, 900, 0.3, 3)
        #potention_well(self, amplitude, a):
        a.potention_well(1300, 0.1)
        #inition_state_psi_well(self, A, a):
        #self.psi_0 = A * np.sin((np.pi * self.grid_x)/a)**3
        a.inition_state_psi_well(100, 0.1)
        #plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, a.V_pote)
        plt.plot(a.grid_x, np.abs(a.psi_0)**2)
        plt.show()
        b = WaveFunctionEvolution(a)
        b.solver(a)
        plt.plot(a.grid_x, a.V_pote)
        #plt.plot(a.grid_x, np.abs(b.psi_x_t[1500-1])**2)
        plt.show()

#z = SimulatedSystems()
#z.free_movement()
#a = ParamsSystem(100, 100, 10, 0.1)
#a.inition_state_psi_gauss(1, 2, -3)
#b = WaveFunctionEvolution(a)
