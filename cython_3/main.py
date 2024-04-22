import numpy as np
import cython
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import random
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
import solver_1d
import solver_2d
import datetime



class Izing_1D():
    def __init__(self, count_partial=225, start_temperatyre = 0.1, end_temperatyre = 4., size_temperature = 400 , iteration = 5000,inition_start = 'cold' ):
        self.J = 1.0
        self.B = 0.
        self.mu = 1.0
        self.matrices_spin_number = np.zeros((count_partial, iteration, size_temperature))
        self.matrices_spin_number_end = np.zeros((count_partial, iteration, size_temperature))
        self.energy = np.zeros((iteration, size_temperature))
        self.energy_T = np.zeros(size_temperature)
        self.inition_conduction(inition_start)
        self.T_0 = start_temperatyre
        self.T_end = end_temperatyre
        self.dT = (end_temperatyre - start_temperatyre)/size_temperature
        self.TT = np.linspace(self.T_0, self.T_end, size_temperature)
        self.size_temperature = int(size_temperature)
        self.magnetization = np.zeros(size_temperature)
        self.heat_condactivity = np.zeros(size_temperature)
        self.temp_solve()
        self.plotter()

    def inition_conduction(self, inition_start):
        if inition_start == 'hot':
            self.matrices_spin_number[:,:,0] = np.random.choice([-1., 1.], size=(self.matrices_spin_number.shape[0]))
        elif inition_start == 'cold':
            self.matrices_spin_number[:,0, :] = 1.
        else:
            print('Нет такого начального условия')
    def temp_solve(self):
        start = datetime.datetime.now()
        print('Время старта: ' + str(start))
        self.matrices_spin_number_end, self.energy = solver_1d.metropolis_start(self.matrices_spin_number,  self.energy, self.TT, self.matrices_spin_number.shape[0], self.matrices_spin_number.shape[1], self.T_0,  self.T_end, self.matrices_spin_number.shape[2], self.J, self.mu, self.B)
        finish = datetime.datetime.now()
        print('Время окончания: ' + str(finish))
        print('Время работы: ' + str(finish - start))
        self.magnetization = solver_1d.average_mag(self.matrices_spin_number_end, self.magnetization, self.matrices_spin_number.shape[2], self.matrices_spin_number.shape[1], self.matrices_spin_number.shape[0])
        self.energy_T = solver_1d.average_e(self.matrices_spin_number_end, self.energy_T, self.matrices_spin_number.shape[2], self.matrices_spin_number.shape[1], self.matrices_spin_number.shape[0], self.J, self.mu, self.B)
        self.heat_condactivity = solver_1d.average_cv(self.matrices_spin_number_end, self.heat_condactivity, self.TT, self.matrices_spin_number.shape[2], self.matrices_spin_number.shape[1], self.matrices_spin_number.shape[0], self.J, self.mu, self.B)

    def plotter(self):
        T = np.linspace(self.T_0, self.T_end, self.size_temperature)
        windows = 30
        for i in range(self.size_temperature - windows-1):
            data = self.magnetization[i:i+windows]
            self.magnetization[i] = np.average(data)
        plt.plot(T[:self.size_temperature - windows], self.magnetization[:self.size_temperature - windows])
        plt.title('Намагниченость')
        plt.xlabel('T')
        plt.ylabel('M')
        plt.show()
        windows = 30
        for i in range(self.size_temperature - windows-1):
            data = self.energy_T[i:i+windows]
            self.energy_T[i] = np.average(data)
        plt.plot(T[:self.size_temperature - windows], self.energy_T[:self.size_temperature - windows]/self.matrices_spin_number_end.shape[0])
        plt.title('Энергия')
        plt.xlabel('T')
        plt.ylabel('U')
        plt.show()
        for i in range(self.size_temperature - windows-1):
            data = self.heat_condactivity[i:i+windows]
            self.heat_condactivity[i] = np.average(data)
        plt.plot(T[:self.size_temperature - windows], self.heat_condactivity[:self.size_temperature - windows]/self.matrices_spin_number_end.shape[0])
        plt.title('Теплоёмкость')
        plt.xlabel('T')
        plt.ylabel('С')
        plt.show()




class Izing_2D():
    def __init__(self, count_partial=15, start_temperatyre = 0.1, end_temperatyre = 4., size_temperature = 400 , iteration = 5000,inition_start = 'cold' ):
        self.J = 1.0
        self.B = 0.
        self.mu = 1.0
        self.matrices_spin_number = np.zeros((count_partial, count_partial, iteration, size_temperature))
        self.matrices_spin_number_end = np.zeros((count_partial, count_partial, iteration, size_temperature))
        self.energy = np.zeros((iteration, size_temperature))
        self.energy_T = np.zeros(size_temperature)
        self.inition_conduction(inition_start)
        self.T_0 = start_temperatyre
        self.T_end = end_temperatyre
        self.dT = (end_temperatyre - start_temperatyre)/size_temperature
        self.TT = np.linspace(self.T_0, self.T_end, size_temperature)
        self.size_temperature = int(size_temperature)
        self.magnetization = np.zeros(size_temperature)
        self.heat_condactivity = np.zeros(size_temperature)
        self.temp_solve()
        self.plotter()

    def inition_conduction(self, inition_start):
        if inition_start == 'hot':
            self.matrices_spin_number[:,:,0] = np.random.choice([-1., 1.], size=(self.matrices_spin_number.shape[0]))
        elif inition_start == 'cold':
            self.matrices_spin_number[:,:,0, :] = 1.
        else:
            print('Нет такого начального условия')
    def temp_solve(self):
        start = datetime.datetime.now()
        print('Время старта: ' + str(start))
        self.matrices_spin_number_end = solver_2d.metropolis_start(self.matrices_spin_number, self.TT, self.matrices_spin_number.shape[0], self.matrices_spin_number.shape[2], self.T_0,  self.T_end, self.matrices_spin_number.shape[3], self.J, self.mu, self.B)
        finish = datetime.datetime.now()
        print('Время окончания: ' + str(finish))
        print('Время работы: ' + str(finish - start))
        self.magnetization = solver_2d.average_mag(self.matrices_spin_number_end, self.magnetization, self.matrices_spin_number.shape[3], self.matrices_spin_number.shape[2], self.matrices_spin_number.shape[0])
        self.energy_T = solver_2d.average_e(self.matrices_spin_number_end, self.energy_T, self.matrices_spin_number.shape[3], self.matrices_spin_number.shape[2], self.matrices_spin_number.shape[0], self.J, self.mu, self.B)
        self.heat_condactivity = solver_2d.average_cv(self.matrices_spin_number_end, self.heat_condactivity, self.TT, self.matrices_spin_number.shape[3], self.matrices_spin_number.shape[2], self.matrices_spin_number.shape[0], self.J, self.mu, self.B)

    def plotter(self):
        T = np.linspace(self.T_0, self.T_end, self.size_temperature)
        windows = 30
        for i in range(self.size_temperature - windows):
            data = self.magnetization[i:i+windows]
            self.magnetization[i] = np.average(data)
        plt.plot(T[:self.size_temperature - windows], self.magnetization[:self.size_temperature - windows])
        plt.title('Намагниченость')
        plt.xlabel('T')
        plt.ylabel('M')
        plt.show()
        windows = 30
        for i in range(self.size_temperature - windows):
            data = self.energy_T[i:i+windows]
            self.energy_T[i] = np.average(data)
        plt.plot(T[:self.size_temperature - windows], self.energy_T[:self.size_temperature - windows]/self.matrices_spin_number_end.shape[0] ** 2)
        plt.title('энергия')
        plt.xlabel('T')
        plt.ylabel('U')
        plt.show()
        for i in range(self.size_temperature - windows):
            data = self.heat_condactivity[i:i+windows]
            self.heat_condactivity[i] = np.average(data)
        plt.plot(T[:self.size_temperature - windows], self.heat_condactivity[:self.size_temperature - windows]/self.matrices_spin_number_end.shape[0])
        plt.title('Теплоёмкость')
        plt.xlabel('T')
        plt.ylabel('C')
        plt.show()


a = Izing_1D()
