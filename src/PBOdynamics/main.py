
import numpy as np
from StochasticMechanics import Stochastic

freq = np.linspace(0, 50, 500)
S0 = np.ones((500, 2))

gamma = np.ones((2,1)) * [0.5]
nu = np.ones((2,1)) * [0.5]
alpha = np.ones((2,1)) * [1]

c=1
k=200
a = 0.01
M = np.array([[1, 0, 0, 0],[1, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])
C = np.array([[c, -c, 0, 0],[0, c, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
K = np.array([[a*k, -a*k, (1-a)*k, -(1-a)*k],[0, a*k, 0, (1-a)*k],[0, 0, 0, 0],[0, 0, 0, 0]])

Sto = Stochastic(power_spectrum_object='white_noise', model='bouc_wen', ndof=2, freq=freq)

Var, Vard = Sto.statistical_linearization(M=M, C=C, K=K, tol=1e-5, maxiter=100, S0=S0, gamma=gamma, nu=nu, alpha=alpha)

print([Var, Vard])