import numpy as np
from numpy.linalg import inv
import scipy as sp
from os import path
import copy
import sys

import matplotlib.pyplot as plt

class Stochastic:

    def __init__(self, power_spectrum_object=None, power_spectrum_script=None, model=None, ndof=None, freq=None):

        # todo: implement the case of full matrix power spectrum
        self.power_spectrum_object = power_spectrum_object
        self.power_spectrum_script = power_spectrum_script
        self.model = model
        self.ndof = ndof
        self.freq = freq

        # Initial guess of the iterative process in the statistical linearization
        if ndof >= 1:
            self.meq0 = np.ones(ndof) * [0]
            self.ceq0 = np.ones(ndof) * [0.5]
            self.keq0 = np.ones(ndof) * [1e3]
        else:
            raise ValueError('ndof MUST be larger than or equal to 1.')

        if power_spectrum_script is not None:
            self.user_ps_check = path.exists(power_spectrum_script)
        else:
            self.user_ps_check = False

        if self.user_ps_check:
            try:
                self.module_dist = __import__(self.power_spectrum_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing a power spectru,.')

        if self.model is 'bouc_wen':
            self.create_matrices = 'create_matrix_bw'
            self.equivalent_elements = 'equivalent_elements_bw'

    def white_noise(self, kwargs):

        if 'S0' in kwargs.keys():
            S0 = kwargs['S0']
        else:
            raise ValueError('S0 cannot be None.')

        power_spectrum = S0

        return np.array(power_spectrum)

    def statistical_linearization(self, M=None, C=None, K=None, tol=1e-3, maxiter=1000, **kwargs):

        if tol<sys.float_info.min:
            raise ValueError('tol cannot be lower than '+str(sys.float_info.min))

        if not isinstance(maxiter,int):
            raise TypeError('maxiter MUST be an integer. ')

        if maxiter<1:
            raise ValueError('maxiter cannot be lower than 1')

        if self.user_ps_check:
            if self.power_spectrum_script is None:
                raise TypeError('power_spectrum_script cannot be None')

            exec('from ' + self.power_spectrum_script[:-3] + ' import ' + self.power_spectrum_object)
            power_spectrum_fun = eval(self.power_spectrum_object)
        else:
            if self.power_spectrum_object is None:
                raise TypeError('power_spectrum_object cannot be None')

            power_spectrum_fun = eval("Stochastic." + self.power_spectrum_object)

        freq = self.freq
        power_spectrum = power_spectrum_fun(self, kwargs)

        Mt = np.zeros(np.shape(M))
        Ct = np.zeros(np.shape(C))
        Kt = np.zeros(np.shape(K))

        meq = copy.copy(self.meq0)
        ceq = copy.copy(self.ceq0)
        keq = copy.copy(self.keq0)

        #meq_1 = copy.copy(self.meq0)/100
        #ceq_1 = copy.copy(self.ceq0)/100
        #keq_1 = copy.copy(self.keq0)/100
        meq_1 = 1e-3
        ceq_1 = 1e-3
        keq_1 = 1e-3

        # Do not use mass as a stop criterion

        Meq, Ceq, Keq = Stochastic.assembly_matrices(self, ceq, keq)
        runs = 1
        error = 1000*tol
        while error > tol and runs <= maxiter:

            Mt = M + Meq
            Ct = C + Ceq
            Kt = K + Keq

            H = []
            for i in range(len(freq)):
                H.append(inv(-(freq[i]**2) * Mt + 1j * freq[i] * Ct + Kt))

            ceq_1 = copy.copy(ceq)
            keq_1 = copy.copy(keq)
            ceq, keq, Var, Vard = Stochastic.update_matrices(self, H, power_spectrum, ceq, keq, kwargs)
            Meq, Ceq, Keq = Stochastic.assembly_matrices(self, ceq, keq)

            dceq = abs(ceq - ceq_1) / abs(ceq_1)
            dkeq = abs(keq - keq_1) / abs(keq_1)

            error = np.min(np.minimum(dceq, dkeq))
            runs = runs + 1

            return Var, Vard

    def assembly_matrices(self, ceq=None, keq=None):

        assembly_matrices_fun = eval("Stochastic." + self.create_matrices)
        Meq, Ceq, Keq = assembly_matrices_fun(self, ceq, keq)

        return Meq, Ceq, Keq

    def create_matrix_bw(self, ceq=None, keq=None):

        ndof = self.ndof
        Meq = np.zeros((2 * ndof, 2 * ndof))
        Ceq = np.zeros((2 * ndof, 2 * ndof))
        Keq = np.zeros((2 * ndof, 2 * ndof))

        cont = 0
        for i in np.arange(ndof,2 * ndof):
            Ceq[i, cont] = ceq[cont]
            Keq[i, i] = keq[cont]
            cont = cont + 1

        return Meq, Ceq, Keq

    def update_matrices(self, H=None, power_spectrum=None, ceq=None, keq=None, kwargs=None):

        update_matrices_fun = eval("Stochastic." + self.equivalent_elements)
        ceq, keq, Var, Vard = update_matrices_fun(self, H, power_spectrum, ceq, keq, kwargs)

        return ceq, keq, Var, Vard

    def equivalent_elements_bw(self, H=None, power_spectrum=None, ceq_in=None, keq_in=None, kwargs=None):

        if 'gamma' in kwargs.keys():
            gamma = kwargs['gamma']
        else:
            raise ValueError('gamma cannot be None.')

        if 'nu' in kwargs.keys():
            nu = kwargs['nu']
        else:
            raise ValueError('nu cannot be None.')

        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            raise ValueError('alpha cannot be None.')

        H_ps = []
        H_ps_freq = []
        aux = np.zeros((len(self.freq), 2*self.ndof))
        for i in range(len(self.freq)):
            aux0 = np.sum(abs(H[i][:,0:self.ndof])**2, axis=1)
            aux[i, 0:self.ndof] = 2*np.diag(power_spectrum[i]).dot(aux0[0:self.ndof])
            aux[i, self.ndof:2*self.ndof] = 2*np.diag(power_spectrum[i]).dot(aux0[self.ndof:2*self.ndof])
            H_ps.append(aux[i])
            H_ps_freq.append((self.freq[i]**2)*aux[i])

        H_ps = np.array(H_ps).T
        H_ps_freq = np.array(H_ps_freq).T

        keq = np.zeros((self.ndof,1))
        ceq = np.zeros((self.ndof, 1))
        Var = np.zeros((self.ndof, 1))
        Vard = np.zeros((self.ndof, 1))
        for i in range(self.ndof):
            Ex = np.trapz(H_ps[i],self.freq)
            Exd = np.trapz(H_ps_freq[i], self.freq)
            Ez = np.trapz(H_ps[i + self.ndof], self.freq)
            Ezd = -(keq_in[i] / ceq_in[i]) * Ez

            Var[i] = Ex
            Vard[i] = Exd
            keq[i] = np.sqrt(2.0/np.pi) * (gamma[i] * np.sqrt(Exd) + nu[i] * Ezd / np.sqrt(Ez))
            ceq[i] = np.sqrt(2.0 / np.pi) * (gamma[i] * Ezd / np.sqrt(Exd) + nu[i] * np.sqrt(Ez)) - alpha[i]

        return ceq, keq, Var, Vard

